//! Phonetic dictionary pipeline for STT correction.
//!
//! Scans STT output with a sliding window (1–3 words) and matches against
//! a pre-built dictionary using:
//! 1. **Phonetic matching** — Double Metaphone encoding (via `rphonetic`)
//! 2. **Fuzzy matching** — Jaro-Winkler similarity (via `strsim`)
//!
//! Candidates are collected and passed to the LLM for final correction,
//! transforming a hard "search" problem into a simple "choose" problem.

use rphonetic::{DoubleMetaphone, Encoder};
use strsim::jaro_winkler;
use tracing::{debug, info};

/// A dictionary term with pre-computed phonetic codes.
#[derive(Debug, Clone)]
struct DictionaryEntry {
    /// The original term as written in the terms file.
    term: String,
    /// Lowercase version for comparison.
    lower: String,
    /// Double Metaphone primary code.
    dm_primary: String,
    /// Double Metaphone alternate code.
    dm_alternate: String,
}

/// Pre-built dictionary for phonetic and fuzzy matching.
#[derive(Debug, Clone)]
pub struct Dictionary {
    entries: Vec<DictionaryEntry>,
    encoder: DoubleMetaphone,
}

/// A candidate correction found by the pipeline.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// The original text span from the STT output (1–3 words).
    pub original: String,
    /// The position range in the word list (start_idx, end_idx exclusive).
    pub word_range: (usize, usize),
    /// The matching dictionary term.
    pub suggested: String,
    /// How this match was found.
    pub match_type: MatchType,
    /// Similarity score (1.0 for exact phonetic match, Jaro-Winkler score for fuzzy).
    pub score: f64,
}

/// How a candidate match was identified.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatchType {
    Phonetic,
    Fuzzy,
}

impl std::fmt::Display for MatchType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatchType::Phonetic => write!(f, "phonetic"),
            MatchType::Fuzzy => write!(f, "fuzzy"),
        }
    }
}

/// Configuration for the dictionary pipeline stages.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub phonetic_enabled: bool,
    pub fuzzy_enabled: bool,
    pub fuzzy_threshold: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            phonetic_enabled: true,
            fuzzy_enabled: true,
            fuzzy_threshold: 0.85,
        }
    }
}

impl Dictionary {
    /// Build a dictionary from terms file content.
    ///
    /// Each non-empty, non-comment line is treated as a term.
    /// Multi-word terms are supported (e.g., "Terraform Cloud").
    pub fn from_terms_content(content: &str) -> Self {
        let encoder = DoubleMetaphone::default();
        let mut entries = Vec::new();

        for line in content.lines() {
            let term = line.trim();
            if term.is_empty() || term.starts_with('#') {
                continue;
            }

            let lower = term.to_lowercase();
            // For multi-word terms, encode the concatenated form
            let joined: String = lower.split_whitespace().collect();
            let dm_result = encoder.double_metaphone(&joined);
            let dm_primary = dm_result.primary();
            let dm_alternate = dm_result.alternate();

            entries.push(DictionaryEntry {
                term: term.to_string(),
                lower,
                dm_primary,
                dm_alternate,
            });
        }

        info!(
            "Dictionary built: {} terms, phonetic codes computed",
            entries.len()
        );
        if !entries.is_empty() {
            debug!(
                "Sample entries: {:?}",
                entries
                    .iter()
                    .take(5)
                    .map(|e| format!(
                        "{} [dm={}|{}]",
                        e.term, e.dm_primary, e.dm_alternate
                    ))
                    .collect::<Vec<_>>()
            );
        }

        Self { entries, encoder }
    }

    /// Returns true if the dictionary has no terms.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Scan STT text and find correction candidates using phonetic and fuzzy matching.
    ///
    /// Uses a sliding window of 1–3 words to catch STT hallucinations that
    /// split a single term into multiple words (e.g., "zinit" → "Z in it").
    pub fn find_candidates(
        &self,
        stt_text: &str,
        config: &PipelineConfig,
    ) -> Vec<Candidate> {
        if self.entries.is_empty() {
            debug!("Dictionary is empty, skipping candidate search");
            return Vec::new();
        }

        if !config.phonetic_enabled && !config.fuzzy_enabled {
            debug!("Both phonetic and fuzzy matching disabled, skipping candidate search");
            return Vec::new();
        }

        let words: Vec<&str> = stt_text.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }

        info!(
            "Scanning {} words with sliding window (1-3), phonetic={}, fuzzy={} (threshold={:.2})",
            words.len(),
            config.phonetic_enabled,
            config.fuzzy_enabled,
            config.fuzzy_threshold,
        );

        let mut all_candidates = Vec::new();

        // Sliding window: 1, 2, and 3 words
        for window_size in 1..=3 {
            if window_size > words.len() {
                break;
            }

            for start in 0..=(words.len() - window_size) {
                let end = start + window_size;
                let window_words = &words[start..end];
                let window_text = window_words.join(" ");
                // For matching: join without spaces (how STT splits relate to real terms)
                let window_joined: String = window_words
                    .iter()
                    .map(|w| w.to_lowercase())
                    .collect::<Vec<_>>()
                    .concat();

                let candidates = self.match_against_dictionary(
                    &window_text,
                    &window_joined,
                    (start, end),
                    config,
                );

                for c in &candidates {
                    debug!(
                        "Window[{}..{}] '{}' -> '{}' ({}, score={:.3})",
                        start, end, c.original, c.suggested, c.match_type, c.score
                    );
                }

                all_candidates.extend(candidates);
            }
        }

        // Deduplicate: if multiple windows match the same dictionary term,
        // keep the one with the highest score
        all_candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Remove candidates where the STT text already matches the term exactly
        all_candidates.retain(|c| {
            let original_lower = c.original.to_lowercase();
            let suggested_lower = c.suggested.to_lowercase();
            original_lower != suggested_lower
        });

        info!(
            "Found {} correction candidates",
            all_candidates.len()
        );

        for c in &all_candidates {
            info!(
                "  Candidate: '{}' -> '{}' ({}, score={:.3})",
                c.original, c.suggested, c.match_type, c.score
            );
        }

        all_candidates
    }

    /// Match a window of text against all dictionary entries.
    fn match_against_dictionary(
        &self,
        original_text: &str,
        joined_lower: &str,
        word_range: (usize, usize),
        config: &PipelineConfig,
    ) -> Vec<Candidate> {
        let mut candidates = Vec::new();

        // Compute phonetic code for the joined window text
        let dm_result = if config.phonetic_enabled {
            Some(self.encoder.double_metaphone(joined_lower))
        } else {
            None
        };
        let window_primary = dm_result.as_ref().map(|r| r.primary()).unwrap_or_default();
        let window_alternate = dm_result.as_ref().map(|r| r.alternate()).unwrap_or_default();

        for entry in &self.entries {
            // Skip entries that are way too different in length
            // (a 3-letter window won't match a 15-letter term)
            let len_ratio = joined_lower.len() as f64 / entry.lower.replace(' ', "").len() as f64;
            if len_ratio < 0.4 || len_ratio > 2.5 {
                continue;
            }

            // Stage 1: Phonetic match (Double Metaphone)
            if config.phonetic_enabled {
                let phonetic_match =
                    (!window_primary.is_empty() && window_primary == entry.dm_primary)
                    || (!window_primary.is_empty() && window_primary == entry.dm_alternate)
                    || (!window_alternate.is_empty() && window_alternate == entry.dm_primary)
                    || (!window_alternate.is_empty() && window_alternate == entry.dm_alternate);

                if phonetic_match {
                    candidates.push(Candidate {
                        original: original_text.to_string(),
                        word_range,
                        suggested: entry.term.clone(),
                        match_type: MatchType::Phonetic,
                        score: 1.0,
                    });
                    continue; // Don't also fuzzy-match the same entry
                }
            }

            // Stage 2: Fuzzy match (Jaro-Winkler)
            if config.fuzzy_enabled {
                let entry_joined: String = entry.lower.split_whitespace().collect();
                let score = jaro_winkler(joined_lower, &entry_joined);
                if score >= config.fuzzy_threshold {
                    candidates.push(Candidate {
                        original: original_text.to_string(),
                        word_range,
                        suggested: entry.term.clone(),
                        match_type: MatchType::Fuzzy,
                        score,
                    });
                }
            }
        }

        candidates
    }
}

/// Format candidates into a concise LLM prompt section.
///
/// Instead of sending the entire dictionary to the LLM, we send only
/// the specific candidates found by the phonetic/fuzzy pipeline. This
/// transforms the problem from "search a 1000-word dictionary" to
/// "choose the right replacement for these specific words".
pub fn format_candidates_for_llm(candidates: &[Candidate]) -> String {
    if candidates.is_empty() {
        return String::new();
    }

    let mut lines = Vec::new();
    lines.push("The following words may be misrecognized. For each, a likely correct term is suggested:".to_string());

    // Group by original text to avoid duplicate entries
    let mut seen = std::collections::HashSet::new();
    for c in candidates {
        let key = format!("{}|{}", c.original.to_lowercase(), c.suggested.to_lowercase());
        if seen.insert(key) {
            lines.push(format!(
                "- \"{}\" might be \"{}\" ({} match, confidence {:.0}%)",
                c.original,
                c.suggested,
                c.match_type,
                c.score * 100.0
            ));
        }
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> PipelineConfig {
        PipelineConfig::default()
    }

    #[test]
    fn test_dictionary_from_terms() {
        let content = "Ansible\nKubernetes\n# comment\nTerraform\n\n";
        let dict = Dictionary::from_terms_content(content);
        assert_eq!(dict.entries.len(), 3);
    }

    #[test]
    fn test_phonetic_match() {
        let dict = Dictionary::from_terms_content("Ansible");
        let candidates = dict.find_candidates("an sible", &test_config());
        assert!(
            candidates.iter().any(|c| c.suggested == "Ansible"),
            "Expected 'an sible' to match 'Ansible', got: {:?}",
            candidates
        );
    }

    #[test]
    fn test_fuzzy_match() {
        let dict = Dictionary::from_terms_content("Kubernetes");
        let candidates = dict.find_candidates("kubernetis", &test_config());
        assert!(
            candidates.iter().any(|c| c.suggested == "Kubernetes"),
            "Expected 'kubernetis' to match 'Kubernetes', got: {:?}",
            candidates
        );
    }

    #[test]
    fn test_trigram_window() {
        let dict = Dictionary::from_terms_content("Zenith");
        let candidates = dict.find_candidates("I like Z in it very much", &test_config());
        assert!(
            candidates.iter().any(|c| c.original == "Z in it" && c.suggested == "Zenith"),
            "Expected 'Z in it' to match 'Zenith', got: {:?}",
            candidates
        );
    }

    #[test]
    fn test_exact_match_filtered() {
        let dict = Dictionary::from_terms_content("Ansible");
        let candidates = dict.find_candidates("Ansible", &test_config());
        // Exact matches should be filtered out
        assert!(
            candidates.is_empty(),
            "Expected no candidates for exact match, got: {:?}",
            candidates
        );
    }

    #[test]
    fn test_disabled_stages() {
        let dict = Dictionary::from_terms_content("Ansible");

        // Both disabled
        let config = PipelineConfig {
            phonetic_enabled: false,
            fuzzy_enabled: false,
            fuzzy_threshold: 0.85,
        };
        let candidates = dict.find_candidates("an sible", &config);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_format_candidates() {
        let candidates = vec![Candidate {
            original: "an sible".to_string(),
            word_range: (0, 2),
            suggested: "Ansible".to_string(),
            match_type: MatchType::Phonetic,
            score: 1.0,
        }];
        let formatted = format_candidates_for_llm(&candidates);
        assert!(formatted.contains("an sible"));
        assert!(formatted.contains("Ansible"));
    }
}
