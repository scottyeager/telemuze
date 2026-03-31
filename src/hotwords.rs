//! Hotwords parsing for sherpa-onnx per-request boosting.
//!
//! Clients send a comma-separated list of words/phrases. This module
//! converts that into the newline-delimited format sherpa-onnx expects,
//! optionally appending per-word `:score` suffixes.

/// Parse a comma-separated hotwords string into sherpa-onnx's
/// newline-delimited format (one word/phrase per line).
///
/// If `score` is provided and > 0, appends ` :score` to each word
/// so sherpa-onnx applies that boost (overriding the global hotwords_score).
pub fn parse_hotwords(input: &str, score: Option<f32>) -> String {
    let suffix = match score {
        Some(s) if s > 0.0 => format!(" :{s}"),
        _ => String::new(),
    };

    input
        .split(',')
        .map(|w| w.trim())
        .filter(|w| !w.is_empty())
        .map(|w| format!("{w}{suffix}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comma_separated_no_score() {
        assert_eq!(parse_hotwords("foo, bar, baz", None), "foo\nbar\nbaz");
    }

    #[test]
    fn comma_separated_with_score() {
        assert_eq!(
            parse_hotwords("foo, bar", Some(2.0)),
            "foo :2\nbar :2"
        );
    }

    #[test]
    fn zero_score_omitted() {
        assert_eq!(parse_hotwords("foo, bar", Some(0.0)), "foo\nbar");
    }

    #[test]
    fn single_word() {
        assert_eq!(parse_hotwords("telemuze", None), "telemuze");
    }

    #[test]
    fn empty_and_whitespace() {
        assert_eq!(parse_hotwords("  , ,hello, , world,", None), "hello\nworld");
    }

    #[test]
    fn already_newline_delimited() {
        assert_eq!(parse_hotwords("foo\nbar", None), "foo\nbar");
    }
}
