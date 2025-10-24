# listener/main.py
# Telemuze Listener - Telegram bot entrypoint
#
# Responsibilities:
# - Authenticate users
# - Accept Telegram audio/video uploads
# - Queue jobs, enforce concurrency
# - Provision short-lived "composer" VMs (via grid3.tfcmd)
# - Transfer files via SSH/SCP, run transcription, fetch results
# - Reply with text (and attach .txt if over Telegram limit)
#
# Notes:
# - TODO: Implement retrieval of composer VM IP/host after deployment
# - TODO: Implement strict host key verification (TOFU) and persistence in known_hosts
# - TODO: Networking/placement specifics when provisioning via grid3.tfcmd
# - TODO: Cancel-on-delete stretch goal is not implemented (use Cancel button)

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import sqlite3
import string
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import asyncssh

# Grid provisioning (assumed available as per user instruction)
from grid3 import tfcmd as grid3_tfcmd
from telegram import (
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    Update,
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ----------------------------
# Configuration and constants
# ----------------------------

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
if not TELEGRAM_BOT_TOKEN:
    print("ERROR: TELEGRAM_BOT_TOKEN is required", file=sys.stderr)
    sys.exit(1)

TELEGRAM_API_ID = os.environ.get("TELEGRAM_API_ID")
TELEGRAM_API_HASH = os.environ.get("TELEGRAM_API_HASH")

# Override composer IP for development
COMPOSER_IP_OVERRIDE = os.environ.get("COMPOSER_IP_OVERRIDE")

# Deployment info for composers
TF_MNEMONIC = os.environ.get("TF_MNEMONIC", "")
if not TF_MNEMONIC:
    print("ERROR: TF_MNEMONIC is required", file=sys.stderr)
    sys.exit(1)

TF_NETWORK = os.environ.get("TF_NETWORK", "main")

TF_NODE_ID = os.environ.get("TF_NODE_ID")
if not TF_NODE_ID:
    print("ERROR: TF_NODE_ID is required", file=sys.stderr)
    sys.exit(1)

# Auth
ALLOWED_USERNAMES = {
    u.strip().lower()
    for u in os.environ.get("ALLOWED_USERNAMES", "").split(",")
    if u.strip()
}
ALLOWED_USER_IDS = {
    int(u)
    for u in os.environ.get("ALLOWED_USER_IDS", "").split(",")
    if u.strip().isdigit()
}

if not ALLOWED_USERNAMES and not ALLOWED_USER_IDS:
    print(
        "ERROR: At least one allowed user is required (ALLOWED_USERNAMES or ALLOWED_USER_IDS)",
        file=sys.stderr,
    )
    sys.exit(1)

COMPOSER_FLIST = os.environ.get(
    "COMPOSER_FLIST",
    "https://hub.threefold.me/scott.3bot/scottyeager-telemuze-composer-latest.flist",
)
COMPOSER_ENTRYPOINT = os.environ.get(
    "COMPOSER_ENTRYPOINT",
    "/sbin/zinit init",
)
COMPOSER_CPUS = int(os.environ.get("COMPOSER_CPUS", "4"))
# Memory in GB
COMPOSER_RAM = int(os.environ.get("COMPOSER_RAM", "8"))
# Root FS size in GB
COMPOSER_ROOTFS = int(os.environ.get("COMPOSER_ROOTFS", "20"))

# Concurrency (TODO: remove. We only support one composer)
MAX_COMPOSERS = int(os.environ.get("MAX_COMPOSERS", "1"))
PER_USER_CONCURRENCY = int(os.environ.get("PER_USER_CONCURRENCY", "1"))

# Defaults
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "turbo")
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "auto")

# Limits/timeouts
JOB_TIMEOUT_SEC = int(os.environ.get("JOB_TIMEOUT_SEC", str(3 * 60 * 60)))  # 3h
FFMPEG_TIMEOUT_SEC = int(os.environ.get("FFMPEG_TIMEOUT_SEC", str(20 * 60)))  # 20m
SSH_CONNECT_TIMEOUT_SEC = int(os.environ.get("SSH_CONNECT_TIMEOUT_SEC", "90"))
SSH_CMD_IDLE_TIMEOUT_SEC = int(os.environ.get("SSH_CMD_IDLE_TIMEOUT_SEC", "300"))

# Interval for warming both the local cache and the remote cache, if applicable
CACHE_WARM_INTERVAL_HOURS = int(os.environ.get("CACHE_WARM_INTERVAL_MIN", "12"))
# Should be enabled if the composer runs on a separate node
CACHE_WARM_DEPLOY = os.environ.get("CACHE_WARM_DEPLOY", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# File and SSH paths
HOME_DIR = Path.home()
STATE_DIR = HOME_DIR / ".telemuze"
STATE_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path(os.environ.get("TELEMUZE_TMP_DIR", "/tmp/telemuze"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

SSH_KEY_OVERRIDE_PATH = os.environ.get("SSH_KEY_OVERRIDE_PATH")
if SSH_KEY_OVERRIDE_PATH:
    SSH_KEY_PATH = Path(SSH_KEY_OVERRIDE_PATH).expanduser().resolve()
else:
    SSH_KEY_PATH = STATE_DIR / "id_ed25519"
SSH_PUB_PATH = Path(f"{SSH_KEY_PATH}.pub")
KNOWN_HOSTS_PATH = STATE_DIR / "known_hosts"
COMPOSER_USERNAME = "root"

# Telegram message char limit
TELEGRAM_TEXT_LIMIT = 4096

LANG_RE = re.compile(r"^[a-z]{2}(-[A-Z]{2})?$")

MODEL_CHOICES = {"tiny", "turbo"}

DB_PATH = STATE_DIR / "db.sqlite"

# ----------------------------
# Logging
# ----------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("telemuze.listener")

# ----------------------------
# User settings storage (SQLite)
# ----------------------------


def _db_init():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            model TEXT NOT NULL,
            language TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def get_user_settings(user_id: int, username: Optional[str]) -> Tuple[str, str]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT model, language FROM user_settings WHERE user_id = ?", (user_id,)
    )
    row = cur.fetchone()
    if row:
        conn.close()
        return row[0], row[1]
    # Insert defaults
    ts = int(time.time())
    conn.execute(
        "INSERT OR REPLACE INTO user_settings (user_id, username, model, language, updated_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, username or "", DEFAULT_MODEL, DEFAULT_LANGUAGE, ts),
    )
    conn.commit()
    conn.close()
    return DEFAULT_MODEL, DEFAULT_LANGUAGE


def set_user_model(user_id: int, username: Optional[str], model: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    ts = int(time.time())
    conn.execute(
        "INSERT OR REPLACE INTO user_settings (user_id, username, model, language, updated_at) VALUES (?, ?, ?, COALESCE((SELECT language FROM user_settings WHERE user_id = ?), ?), ?)",
        (user_id, username or "", model, user_id, DEFAULT_LANGUAGE, ts),
    )
    conn.commit()
    conn.close()


def set_user_language(user_id: int, username: Optional[str], language: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    ts = int(time.time())
    conn.execute(
        "INSERT OR REPLACE INTO user_settings (user_id, username, model, language, updated_at) VALUES (?, ?, COALESCE((SELECT model FROM user_settings WHERE user_id = ?), ?), ?, ?)",
        (user_id, username or "", user_id, DEFAULT_MODEL, language, ts),
    )
    conn.commit()
    conn.close()


# ----------------------------
# SSH key management
# ----------------------------


def ensure_ssh_keypair():
    if SSH_KEY_OVERRIDE_PATH:
        if not SSH_KEY_PATH.exists():
            log.critical("SSH key override path %s does not exist.", SSH_KEY_PATH)
            sys.exit(1)
        if not SSH_PUB_PATH.exists():
            log.warning(
                "SSH public key %s not found for override key. Deployment may fail if needed.",
                SSH_PUB_PATH,
            )
        return

    if SSH_KEY_PATH.exists() and SSH_PUB_PATH.exists():
        return
    log.info("Generating SSH keypair at %s", SSH_KEY_PATH)
    SSH_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Use ssh-keygen to create an ed25519 key
    subprocess.run(
        [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-N",
            "",
            "-f",
            str(SSH_KEY_PATH),
            "-C",
            "telemuze",
        ],
        check=True,
    )


# ----------------------------
# Job and scheduler
# ----------------------------


@dataclass
class Job:
    job_id: str
    user_id: int
    username: Optional[str]
    chat_id: int
    original_message_id: int
    local_input_path: Path
    original_filename: str
    model: str
    language: str
    status_message_id: Optional[int] = None
    preliminary_message_id: Optional[int] = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    # runtime fields
    vm_name: str = ""
    vm_ip: Optional[str] = None


class Scheduler:
    def __init__(self):
        self.queue: asyncio.Queue[Job] = asyncio.Queue()
        self.global_sem = asyncio.Semaphore(MAX_COMPOSERS)
        self.user_sems: Dict[int, asyncio.Semaphore] = {}
        self.jobs: Dict[str, Job] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        self.last_cache_warm_ts = 0
        self._stop_event = asyncio.Event()

    def get_user_sem(self, user_id: int) -> asyncio.Semaphore:
        if user_id not in self.user_sems:
            self.user_sems[user_id] = asyncio.Semaphore(PER_USER_CONCURRENCY)
        return self.user_sems[user_id]

    def queue_position(self) -> int:
        return self.queue.qsize() + 1

    async def submit(self, job: Job):
        self.jobs[job.job_id] = job
        await self.queue.put(job)

    async def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        job.cancel_event.set()
        # If job is still in queue, it's difficult to remove from asyncio.Queue without draining.
        # We'll let the scheduler skip it when popped.
        return True

    async def run(self, app: Application):
        while not self._stop_event.is_set():
            job: Job = await self.queue.get()
            if job.cancel_event.is_set():
                await self._set_status(app, job, "Canceled")
                self._cleanup_local(job)
                self.jobs.pop(job.job_id, None)
                self.queue.task_done()
                continue

            # Acquire semaphores
            await self.global_sem.acquire()
            user_sem = self.get_user_sem(job.user_id)
            await user_sem.acquire()

            # Start task
            task = asyncio.create_task(self._run_job(app, job, user_sem))
            self.tasks[job.job_id] = task
            self.queue.task_done()

    async def shutdown(self):
        self._stop_event.set()
        # Do not cancel running jobs abruptly; allow graceful stop.

    async def _set_status(self, app: Application, job: Job, text: string):
        if job.status_message_id is None:
            return
        try:
            await app.bot.edit_message_text(
                chat_id=job.chat_id,
                message_id=job.status_message_id,
                text=f"{text}",
            )
        except Exception as e:
            log.warning("Failed to edit status message: %s", e)

    def _cleanup_local(self, job: Job):
        with contextlib.suppress(Exception):
            if job.local_input_path.exists():
                job.local_input_path.unlink()
            job.local_input_path.parent.rmdir()

    async def _run_job(self, app: Application, job: Job, user_sem: asyncio.Semaphore):
        try:
            # Provision composer
            await self._set_status(app, job, "Provisioning worker…")
            vm_name = f"cmp{job.job_id[:8]}"
            job.vm_name = vm_name

            vm_ip = await provision_composer(vm_name)
            log.info("VM %s provisioned with IP %s", vm_name, vm_ip)
            job.vm_ip = vm_ip

            if job.cancel_event.is_set():
                await self._set_status(app, job, "Canceled")
                return

            # Connect via SSH (retry while VM boots)
            await self._set_status(app, job, "Connecting to worker…")
            conn = await connect_ssh_with_retries(vm_ip)

            try:
                # Prepare remote dirs
                remote_input_dir = f"/job/input/{job.job_id}"
                remote_output_dir = f"/job/output/{job.job_id}"
                remote_logs_dir = "/job/logs"
                await run_ssh_command(
                    conn,
                    f"mkdir -p {remote_input_dir} {remote_output_dir} {remote_logs_dir}",
                )

                # Upload file
                await self._set_status(app, job, "Uploading...")
                remote_input_path = (
                    f"{remote_input_dir}/{sanitize_filename(job.original_filename)}"
                )
                await asyncssh.scp(
                    str(job.local_input_path),
                    (conn, remote_input_path),
                    recurse=False,
                    preserve=True,
                )

                if job.cancel_event.is_set():
                    await self._set_status(app, job, "Canceled")
                    return

                # Run transcription
                await self._set_status(app, job, "Transcribing…")
                cmd = (
                    f"/usr/bin/uv run /opt/telemuze/composer.py "
                    f"--in {sh_quote(remote_input_path)} "
                    f"--model {sh_quote(job.model)} "
                    f"--language {sh_quote(job.language)} "
                    f"--job-id {sh_quote(job.job_id)}"
                )
                # Wrap with timeout
                json_result = await run_ssh_command_with_json(
                    conn, cmd, timeout=JOB_TIMEOUT_SEC
                )

                if job.cancel_event.is_set():
                    await self._set_status(app, job, "Canceled")
                    return

                if not json_result.get("ok"):
                    err = json_result.get("error", "Transcription failed")
                    await self._send_error_reply(
                        app, job, f"Transcription failed: {err}"
                    )
                    return

                text_path = json_result.get("text_path")
                if not text_path:
                    await self._send_error_reply(app, job, "No transcript produced.")
                    return

                # Fetch transcript text via cat
                transcript_text = await run_ssh_command(
                    conn,
                    f"cat {sh_quote(text_path)}",
                    timeout=SSH_CMD_IDLE_TIMEOUT_SEC,
                )

                # Send transcript to Telegram
                await self._send_transcript_reply(app, job, transcript_text)

                await self._set_status(app, job, "Done ✅")
            finally:
                # Cleanup remote files
                with contextlib.suppress(Exception):
                    await run_ssh_command(
                        conn,
                        f"rm -rf /job/input/{job.job_id} /job/output/{job.job_id}",
                        timeout=60,
                    )
                conn.close()

        except asyncio.TimeoutError:
            await self._send_error_reply(
                app,
                job,
                "This job exceeded the maximum processing time and was canceled.",
            )
        except Exception as e:
            log.exception("Job %s failed: %s", job.job_id, e)
            await self._send_error_reply(
                app, job, "An internal error occurred while processing your file."
            )
        finally:
            self._cleanup_local(job)
            if job.vm_name:
                await destroy_composer(job.vm_name)

            self.tasks.pop(job.job_id, None)
            self.jobs.pop(job.job_id, None)
            self.global_sem.release()
            user_sem.release()

    async def _send_error_reply(self, app: Application, job: Job, text: str):
        full_text = f"Failed ❌\n{text}"
        if job.status_message_id:
            try:
                await app.bot.edit_message_text(
                    chat_id=job.chat_id,
                    message_id=job.status_message_id,
                    text=full_text,
                )
            except Exception as e:
                log.warning("Failed to edit status message for error: %s", e)
        else:
            # Fallback if there is no status message for some reason
            try:
                await app.bot.send_message(
                    chat_id=job.chat_id,
                    text=full_text,
                    reply_to_message_id=job.original_message_id,
                )
            except Exception as e:
                log.warning("Failed to send error reply: %s", e)

    async def _send_transcript_reply(
        self, app: Application, job: Job, transcript_text: str
    ):
        text = transcript_text.strip()
        if not text:
            # If there's a preliminary message, edit it. Otherwise, send a new one.
            if job.preliminary_message_id:
                await app.bot.edit_message_text(
                    chat_id=job.chat_id,
                    message_id=job.preliminary_message_id,
                    text="ℹ️ Transcription completed, but no speech was detected.",
                )
            else:
                await app.bot.send_message(
                    chat_id=job.chat_id,
                    text="ℹ️ Transcription completed, but no speech was detected.",
                    reply_to_message_id=job.original_message_id,
                )
            return

        # If we have a preliminary message, we edit it.
        if job.preliminary_message_id:
            if len(text) <= TELEGRAM_TEXT_LIMIT:
                await app.bot.edit_message_text(
                    chat_id=job.chat_id,
                    message_id=job.preliminary_message_id,
                    text=text,
                )
            else:
                # Send truncated text in the edited message
                truncated = text[:TELEGRAM_TEXT_LIMIT]
                await app.bot.edit_message_text(
                    chat_id=job.chat_id,
                    message_id=job.preliminary_message_id,
                    text=truncated,
                )
                # Attach full transcript as a new message, replying to the original
                with tempfile.NamedTemporaryFile(
                    "w+", encoding="utf-8", delete=False, suffix=".txt"
                ) as tf:
                    tf.write(text)
                    tmp_path = tf.name
                try:
                    await app.bot.send_document(
                        chat_id=job.chat_id,
                        document=InputFile(
                            tmp_path, filename=f"transcript-{job.job_id}.txt"
                        ),
                        caption="Full transcript",
                        reply_to_message_id=job.original_message_id,
                    )
                finally:
                    with contextlib.suppress(Exception):
                        os.unlink(tmp_path)
        else:
            # Standard behavior: send as a new message
            if len(text) <= TELEGRAM_TEXT_LIMIT:
                await app.bot.send_message(
                    chat_id=job.chat_id,
                    text=text,
                    reply_to_message_id=job.original_message_id,
                )
            else:
                truncated = text[:TELEGRAM_TEXT_LIMIT]
                await app.bot.send_message(
                    chat_id=job.chat_id,
                    text=truncated,
                    reply_to_message_id=job.original_message_id,
                )
                with tempfile.NamedTemporaryFile(
                    "w+", encoding="utf-8", delete=False, suffix=".txt"
                ) as tf:
                    tf.write(text)
                    tmp_path = tf.name
                try:
                    await app.bot.send_document(
                        chat_id=job.chat_id,
                        document=InputFile(
                            tmp_path, filename=f"transcript-{job.job_id}.txt"
                        ),
                        caption="Full transcript",
                        reply_to_message_id=job.original_message_id,
                    )
                finally:
                    with contextlib.suppress(Exception):
                        os.unlink(tmp_path)


async def provision_composer(vm_name: str) -> str:
    """
    Provision a composer VM and return its IP address or hostname.

    TODO:
    - Pass placement hints (TF_NODE_ID/TF_FARM_ID) to tfcmd
    - Inject the SSH public key into the VM's authorized_keys
    - Retrieve the assigned public IP or DNS name
    """
    if COMPOSER_IP_OVERRIDE:
        log.info("Using override composer IP: %s", COMPOSER_IP_OVERRIDE)
        return COMPOSER_IP_OVERRIDE

    log.info("Provisioning VM %s", vm_name)
    vm_info = await asyncio.to_thread(
        tfcmd.deploy_vm,
        vm_name,
        ssh=str(SSH_PUB_PATH),
        node=TF_NODE_ID,
        flist=COMPOSER_FLIST,
        cpu=COMPOSER_CPUS,
        memory=COMPOSER_RAM,
        rootfs=COMPOSER_ROOTFS,
        entrypoint=COMPOSER_ENTRYPOINT,
    )
    vm_ip = vm_info.get("mycelium_ip")
    if not vm_ip:
        raise RuntimeError(f"No mycelium_ip returned for VM {vm_name}")

    return vm_ip


async def destroy_composer(vm_name: str):
    if COMPOSER_IP_OVERRIDE:
        log.info("Skipping VM destruction for override composer")
        return

    log.info("Destroying VM %s", vm_name)
    with contextlib.suppress(Exception):
        await asyncio.to_thread(tfcmd.cancel_vm, vm_name)


# ----------------------------
# SSH helpers
# ----------------------------


async def connect_ssh_with_retries(host: str) -> asyncssh.SSHClientConnection:
    start = time.time()
    last_err: Optional[Exception] = None
    while time.time() - start < SSH_CONNECT_TIMEOUT_SEC:
        try:
            conn = await asyncssh.connect(
                host=host,
                username=COMPOSER_USERNAME,
                client_keys=[str(SSH_KEY_PATH)],
                known_hosts=None,
            )
            return conn
        except Exception as e:
            last_err = e
            await asyncio.sleep(3)
    raise RuntimeError(f"SSH connect timeout to {host}: {last_err}")


async def run_ssh_command(
    conn: asyncssh.SSHClientConnection, cmd: str, timeout: Optional[int] = None
) -> str:
    log.debug("SSH RUN: %s", cmd)
    async with conn.create_process(cmd) as proc:
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                proc.kill()
            raise
        if proc.exit_status != 0:
            raise RuntimeError(
                f"Remote command failed ({proc.exit_status}): {stderr or stdout}"
            )
        return stdout


async def run_ssh_command_with_json(
    conn: asyncssh.SSHClientConnection, cmd: str, timeout: Optional[int] = None
) -> dict:
    out = await run_ssh_command(conn, cmd, timeout=timeout)
    # transcribe.sh prints one JSON line at the end
    try:
        # Use last non-empty line as JSON
        lines = [l for l in out.splitlines() if l.strip()]
        return json.loads(lines[-1]) if lines else {}
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from output: {e}\nOut: {out[:5000]}")


def sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return safe[:128] if len(safe) > 128 else safe


# ----------------------------
# Telegram handlers
# ----------------------------


def is_user_allowed(username: Optional[str], user_id: int) -> bool:
    if ALLOWED_USER_IDS and user_id in ALLOWED_USER_IDS:
        return True
    if ALLOWED_USERNAMES and (username or "") in ALLOWED_USERNAMES:
        return True
    return False


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user:
        return
    if not is_user_allowed(update.effective_user.username, update.effective_user.id):
        await update.effective_message.reply_text("Access denied.")
        return
    await update.effective_message.reply_text(
        "Hi! Send me an audio or video, and I’ll transcribe it.\n"
        "Commands:\n"
        "/model <tiny|base|small|medium|large-v3|turbo>\n"
        "/language <auto|en|es|de|...>\n"
        "/settings to view your current settings."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start_cmd(update, context)


async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        return
    if not is_user_allowed(user.username, user.id):
        await update.effective_message.reply_text("Access denied.")
        return
    model, lang = get_user_settings(user.id, user.username)
    await update.effective_message.reply_text(
        f"Your settings:\n- Model: {model}\n- Language: {lang}"
    )


async def model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        return
    if not is_user_allowed(user.username, user.id):
        await update.effective_message.reply_text("Access denied.")
        return
    if not context.args:
        await update.effective_message.reply_text(
            f"Usage: /model <{'|'.join(MODEL_CHOICES)}>"
        )
        return
    model = context.args[0].strip().lower()
    if model not in MODEL_CHOICES:
        await update.effective_message.reply_text(
            f"Invalid model. Choose one of: {', '.join(MODEL_CHOICES)}"
        )
        return
    set_user_model(user.id, user.username, model)
    await update.effective_message.reply_text(f"Model set to: {model}")


async def language_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        return
    if not is_user_allowed(user.username, user.id):
        await update.effective_message.reply_text("Access denied.")
        return
    if not context.args:
        await update.effective_message.reply_text("Usage: /language <auto|code>")
        return
    lang = context.args[0].strip().lower()
    if lang != "auto" and not LANG_RE.match(lang):
        await update.effective_message.reply_text(
            "Invalid language code. Use 'auto' or ISO 639-1 codes like en, es, de."
        )
        return
    set_user_language(user.id, user.username, lang)
    await update.effective_message.reply_text(f"Language set to: {lang}")


def build_cancel_keyboard(job_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Cancel", callback_data=f"cancel:{job_id}")]]
    )


async def cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    m = re.match(r"^cancel:(?P<job_id>[\w-]+)$", query.data)
    if not m:
        return
    job_id = m.group("job_id")
    job = scheduler.jobs.get(job_id)
    user = update.effective_user
    if not job or not user or user.id != job.user_id:
        # Only owner can cancel
        return
    ok = await scheduler.cancel_job(job_id)
    if ok:
        with contextlib.suppress(Exception):
            await query.edit_message_text("Canceling…")
    else:
        with contextlib.suppress(Exception):
            await query.edit_message_text("Unable to cancel.")


async def run_local_transcription(job: Job) -> Optional[Path]:
    """Run composer.py locally for a quick first pass."""
    try:
        # We need to set environment variables to override the default /job/* paths
        # since we're running this on the listener VM, not in a composer.
        local_job_dir = TMP_DIR / job.job_id
        local_out_dir = local_job_dir / "output"
        local_log_dir = local_job_dir / "logs"
        local_out_dir.mkdir(parents=True, exist_ok=True)
        local_log_dir.mkdir(parents=True, exist_ok=True)

        # The composer.py script expects to be run from the project root.
        project_root = Path(__file__).parent.resolve()
        composer_script = project_root / "composer.py"

        cmd = [
            sys.executable,
            str(composer_script),
            "--in",
            str(job.local_input_path),
            "--model",
            "tiny",  # Always use tiny for the fast pass
            "--language",
            job.language,
            "--job-id",
            job.job_id,
        ]

        env = os.environ.copy()
        env["JOB_ID"] = job.job_id
        env["OUT_ROOT"] = str(local_out_dir)
        env["LOG_DIR"] = str(local_log_dir)
        # We don't need to override IN_ROOT as the full path is passed.

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            log.error(
                "Local transcription failed for job %s: %s",
                job.job_id,
                stderr.decode(),
            )
            return None

        # The last line of stdout should be the JSON result
        lines = stdout.decode().strip().splitlines()
        if not lines:
            log.error("Local transcription produced no output for job %s", job.job_id)
            return None

        result = json.loads(lines[-1])
        if result.get("ok") and result.get("text_path"):
            return Path(result["text_path"])
        else:
            log.error(
                "Local transcription failed for job %s: %s",
                job.job_id,
                result.get("error", "Unknown error"),
            )
            return None

    except Exception as e:
        log.exception("Error running local transcription for job %s: %s", job.job_id, e)
        return None


async def _update_preliminary_transcript(
    app: Application, job: Job, preliminary_msg_id: int
):
    """Task to run local transcription and update the preliminary message."""
    transcript_path = await run_local_transcription(job)
    if transcript_path and transcript_path.exists():
        with open(transcript_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            # Note: We don't do the full text splitting logic here,
            # just send the preliminary text, truncated if necessary.
            # The final reply will handle it properly.
            if len(text) > TELEGRAM_TEXT_LIMIT:
                text = text[: TELEGRAM_TEXT_LIMIT - 3] + "..."
            await app.bot.edit_message_text(
                chat_id=job.chat_id,
                message_id=preliminary_msg_id,
                text=f"**Preliminary Transcript (tiny model):**\n\n{text}",
            )
        else:
            await app.bot.edit_message_text(
                chat_id=job.chat_id,
                message_id=preliminary_msg_id,
                text="ℹ️ Preliminary transcript was empty.",
            )


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    user = update.effective_user
    if not user:
        return
    if not is_user_allowed(user.username, user.id):
        await message.reply_text("Access denied.")
        return

    tg_file_id, inferred_name = extract_file_id_and_name(message)
    if not tg_file_id:
        await message.reply_text(
            "I can’t process that message. Please send an audio or video file."
        )
        return

    model, lang = get_user_settings(user.id, user.username)
    job_id = str(uuid.uuid4())
    job_dir = TMP_DIR / job_id / "input"
    job_dir.mkdir(parents=True, exist_ok=True)
    local_path = job_dir / inferred_name

    status_msg = await message.reply_text(
        "Downloading...", reply_to_message_id=message.message_id
    )
    try:
        tg_file = await context.bot.get_file(tg_file_id)
        await tg_file.download_to_drive(custom_path=str(local_path))
    except Exception as e:
        log.exception("Download failed: %s", e)
        await status_msg.edit_text("Failed to download the file from Telegram.")
        return

    job = Job(
        job_id=job_id,
        user_id=user.id,
        username=user.username,
        chat_id=message.chat_id,
        original_message_id=message.message_id,
        local_input_path=local_path,
        original_filename=inferred_name,
        model=model,
        language=lang,
        status_message_id=status_msg.message_id,
    )

    # --- Two-phase transcription logic ---
    preliminary_task = None
    if model == "turbo":
        preliminary_msg = await message.reply_text(
            "Generating preliminary transcript (tiny model)...",
            reply_to_message_id=message.message_id,
        )
        job.preliminary_message_id = preliminary_msg.message_id
        preliminary_task = asyncio.create_task(
            _update_preliminary_transcript(
                context.application, job, preliminary_msg.message_id
            )
        )

    # We always queue the main job, regardless of whether a preliminary one was started.
    pos = scheduler.queue_position()
    await status_msg.edit_text(
        f"Queued (position {pos})",
        reply_markup=build_cancel_keyboard(job_id),
    )
    await scheduler.submit(job)

    if preliminary_task:
        # We can optionally wait for the preliminary task to finish here if we want to
        # ensure it completes before the function returns, but it's not strictly necessary.
        pass


def extract_file_id_and_name(message) -> Tuple[Optional[str], str]:
    # Voice note (OGG Opus)
    if message.voice:
        return message.voice.file_id, f"voice-{message.id or int(time.time())}.ogg"
    # Audio (music/podcast)
    if message.audio:
        name = message.audio.file_name or f"audio-{message.id or int(time.time())}.mp3"
        return message.audio.file_id, name
    # Video
    if message.video:
        name = message.video.file_name or f"video-{message.id or int(time.time())}.mp4"
        return message.video.file_id, name
    # VideoNote (round video)
    if message.video_note:
        return (
            message.video_note.file_id,
            f"videonote-{message.id or int(time.time())}.mp4",
        )
    # Document (generic)
    if message.document:
        name = (
            message.document.file_name or f"document-{message.id or int(time.time())}"
        )
        return message.document.file_id, name
    return None, ""


# ----------------------------
# Cache warmer
# ----------------------------


async def cache_warmer_task(app: Application):
    """
    Periodically spin a composer to keep the composer flist cached on the node. Activity timestamp is also reset by regular composer deployments.
    """
    while True:
        try:
            now = time.time()
            idle_sec = now - scheduler.last_cache_warm_ts
            if (
                idle_sec > CACHE_WARM_INTERVAL_HOURS * 60 * 60
                and scheduler.queue.empty()
            ):
                vm_name = f"cmpwrm{int(now)}"
                log.info("Running cache warmer: %s", vm_name)
                if CACHE_WARM_DEPLOY:
                    try:
                        vm_ip = await provision_composer(vm_name)
                        conn = await connect_ssh_with_retries(vm_ip)
                        try:
                            await run_ssh_command(
                                conn,
                                "/usr/local/bin/reset-model-atimes.sh",
                                timeout=120,
                            )
                        finally:
                            conn.close()
                    except Exception as e:
                        log.warning("Warm provision failed: %s", e)
                        await destroy_composer(vm_name)
                        await asyncio.sleep(60)
                        continue
                    await destroy_composer(vm_name)
                else:
                    try:
                        subprocess.run(
                            ["/usr/local/bin/reset-model-atimes.sh"], check=True
                        )
                    except Exception as e:
                        log.warning("Local reset-model-atimes.sh failed: %s", e)
                # Reset last activity to now to avoid back-to-back warmers
                scheduler.last_cache_warm_ts = time.time()
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning("Cache warmer error: %s", e)
            await asyncio.sleep(60)


# ----------------------------
# Application lifecycle
# ----------------------------


async def cleanup_leftovers():
    """Cleans up leftover VMs and temporary files from previous runs."""
    # Clean up old temp files
    log.info("Cleaning up temporary file directory: %s", TMP_DIR)
    try:
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR)
        TMP_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log.error("Failed to clean up temporary directory %s: %s", TMP_DIR, e)

    # Clean up old VMs
    if COMPOSER_IP_OVERRIDE:
        log.info("Composer IP override set; skipping VM cleanup.")
        return

    log.info("Searching for leftover composer VMs...")
    try:
        # Assumes tfcmd.list() returns a list of deployment names
        all_deployments = await asyncio.to_thread(tfcmd.list)
        composer_vms = [name for name in all_deployments if name.startswith("cmp")]

        if not composer_vms:
            log.info("No leftover composer VMs found.")
            return

        log.info(
            "Found %d leftover composer VMs: %s",
            len(composer_vms),
            ", ".join(composer_vms),
        )
        cleanup_tasks = [destroy_composer(vm_name) for vm_name in composer_vms]
        results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        for vm_name, result in zip(composer_vms, results):
            if isinstance(result, Exception):
                log.error("Failed to destroy leftover VM %s: %s", vm_name, result)
            else:
                log.info("Cleaned up leftover VM: %s", vm_name)

    except AttributeError:
        log.error(
            "Failed to clean up leftover VMs: `tfcmd.list` method not found. Please check grid3.tfcmd library version."
        )
    except Exception as e:
        log.error("Failed to clean up leftover VMs: %s", e)


async def on_startup(app: Application):
    # Disabled until we can check the implementation
    await cleanup_leftovers()

    _db_init()
    ensure_ssh_keypair()
    log.info("Allowed usernames: %s", ", ".join(sorted(ALLOWED_USERNAMES)))
    log.info("Allowed user IDs: %s", ", ".join(map(str, sorted(ALLOWED_USER_IDS))))

    # Start scheduler
    app.bot_data["scheduler_task"] = asyncio.create_task(scheduler.run(app))
    # Start cache warmer
    if COMPOSER_IP_OVERRIDE:
        log.info("Composer IP override set; cache warmer disabled.")
    else:
        app.bot_data["warmer_task"] = asyncio.create_task(cache_warmer_task(app))


async def on_shutdown(app: Application):
    await scheduler.shutdown()
    # Cancel background tasks
    for key in ("scheduler_task", "warmer_task"):
        task: Optional[asyncio.Task] = app.bot_data.get(key)
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def build_application() -> Application:
    builder = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN)

    # Local bot API server support
    if TELEGRAM_API_ID and TELEGRAM_API_HASH:
        log.info("Using local Telegram Bot API server")
        builder.base_url("http://127.0.0.1:8081/bot")
        builder.base_file_url("http://127.0.0.1:8081/file/bot")
        builder.local_mode(True)
        builder.read_timeout(300)

    app = builder.post_init(on_startup).post_shutdown(on_shutdown).build()

    # Commands
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("settings", settings_cmd))
    app.add_handler(CommandHandler("model", model_cmd))
    app.add_handler(CommandHandler("language", language_cmd))

    # Cancel button
    app.add_handler(CallbackQueryHandler(cancel_callback, pattern=r"^cancel:"))

    # Media handler
    media_filter = (
        filters.VOICE
        | filters.AUDIO
        | filters.VIDEO
        | filters.VIDEO_NOTE
        | filters.Document.ALL  # We'll validate in handler if needed
    )
    app.add_handler(MessageHandler(media_filter, handle_media))

    return app


async def logout_from_public_api():
    """
    If using a local bot API, we must first log out from the public cloud API.
    """
    log.info("Deregistering from public Telegram cloud API...")
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        if await bot.log_out():
            log.info("Successfully deregistered from cloud API.")
        else:
            # This can happen if already logged out, not necessarily an error.
            log.warning("Failed to deregister from cloud API (maybe already done).")
    except Exception as e:
        log.error("Error deregistering from cloud API: %s", e)


def main():
    global scheduler, tfcmd
    scheduler = Scheduler()

    tfcmd = grid3_tfcmd.TFCmd()
    tfcmd.login(TF_MNEMONIC)

    # If using local API server, log out from cloud first
    if TELEGRAM_API_ID and TELEGRAM_API_HASH:
        asyncio.run(logout_from_public_api())

    app = build_application()
    # Graceful shutdown on SIGINT/SIGTERM
    try:
        app.run_polling(close_loop=False, allowed_updates=Update.ALL_TYPES)
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
