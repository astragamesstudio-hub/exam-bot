import os
import json
import logging
import threading
import requests
import pdfplumber
from flask import Flask, request as flask_request
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from groq import Groq

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  Config  (all values from environment secrets)
# ─────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_BOT_TOKEN"]
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]
RAZORPAY_KEY     = os.environ["RAZORPAY_KEY"]
RAZORPAY_SECRET  = os.environ["RAZORPAY_SECRET"]

GROQ_MODEL       = "llama-3.3-70b-versatile"
MAX_PDF_CHARS    = 7000
TELEGRAM_LIMIT   = 4000
FREE_LIMIT       = 5
WEBHOOK_PORT     = 8080          # separate from the Node.js server on 5000
DATA_FILE        = "data.json"

SYSTEM_PROMPT = (
    "You are an expert exam analyst and academic coach with years of experience predicting "
    "exam questions. You deeply analyse syllabus content and past exam papers to identify "
    "high-probability topics. Your predictions are accurate, specific, and well-reasoned. "
    "Always prioritise topics with the highest exam weightage and historical frequency."
)

# ─────────────────────────────────────────────
#  Shared state  (protected by a lock because
#  Flask thread and bot thread both access it)
# ─────────────────────────────────────────────
_lock       = threading.Lock()
paid_users  = set()      # set of int user_ids
free_usage  = {}         # {user_id: int count}
user_state  = {}         # {user_id: "syllabus" | "paper" | "done"}

groq_client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────
#  Persistence
# ─────────────────────────────────────────────
def save_data() -> None:
    with _lock:
        snapshot = {"paid": list(paid_users), "free": free_usage}
    with open(DATA_FILE, "w") as fh:
        json.dump(snapshot, fh)


def load_data() -> None:
    global paid_users, free_usage
    try:
        with open(DATA_FILE) as fh:
            data = json.load(fh)
        with _lock:
            paid_users = set(data.get("paid", []))
            free_usage = {int(k): v for k, v in data.get("free", {}).items()}
        logger.info("Data loaded successfully.")
    except FileNotFoundError:
        logger.info("No existing data file — starting fresh.")
    except Exception as exc:
        logger.warning(f"Could not load data (starting fresh): {exc}")


load_data()

# ─────────────────────────────────────────────
#  Payment helpers
# ─────────────────────────────────────────────
def create_razorpay_link(chat_id: int) -> str:
    resp = requests.post(
        "https://api.razorpay.com/v1/payment_links",
        auth=(RAZORPAY_KEY, RAZORPAY_SECRET),
        json={
            "amount": 2900,
            "currency": "INR",
            "description": "Exam Bot – Unlimited Access",
            "notes": {"telegram_id": str(chat_id)},
        },
        timeout=10,
    )
    data = resp.json()
    if "short_url" not in data:
        raise RuntimeError(f"Razorpay error: {data}")
    return data["short_url"]


def notify_user_via_api(user_id: int, text: str) -> None:
    """Send a Telegram message using the HTTP API directly (thread-safe, no asyncio needed)."""
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": user_id, "text": text},
            timeout=10,
        )
    except Exception as exc:
        logger.error(f"Failed to notify user {user_id}: {exc}")

# ─────────────────────────────────────────────
#  User state helpers
# ─────────────────────────────────────────────
def is_paid(user_id: int) -> bool:
    with _lock:
        return user_id in paid_users


def get_remaining(user_id: int) -> int:
    with _lock:
        return max(0, FREE_LIMIT - free_usage.get(user_id, 0))


def increment_usage(user_id: int) -> None:
    with _lock:
        free_usage[user_id] = free_usage.get(user_id, 0) + 1
    save_data()

# ─────────────────────────────────────────────
#  Telegram helpers
# ─────────────────────────────────────────────
async def send_long(update: Update, text: str) -> None:
    """Split a long message into Telegram-sized chunks."""
    for i in range(0, len(text), TELEGRAM_LIMIT):
        await update.message.reply_text(text[i:i + TELEGRAM_LIMIT])


async def show_paywall(update: Update, user_id: int) -> None:
    try:
        link = create_razorpay_link(user_id)
        keyboard = [[InlineKeyboardButton("💳 Pay ₹29 – Unlock Unlimited", url=link)]]
        await update.message.reply_text(
            "🚫 You have used all 5 free generations.\n\n"
            "Pay ₹29 once for unlimited exam predictions forever! 🎓",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
    except Exception as exc:
        logger.error(f"Paywall error for user {user_id}: {exc}")
        await update.message.reply_text(
            "You have reached the free limit. "
            "Payment system is temporarily unavailable — please try again later."
        )


async def ask_groq(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=2048,
        temperature=0.4,
    )
    return response.choices[0].message.content

# ─────────────────────────────────────────────
#  Bot command / message handlers
# ─────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user.id
    logger.info(f"[/start] user={user}")
    user_state[user] = "syllabus"
    context.user_data.clear()

    remaining = get_remaining(user)
    status = (
        "✅ Unlimited access active."
        if is_paid(user)
        else f"🆓 {remaining} free generation(s) remaining."
    )
    await update.message.reply_text(
        f"👋 Welcome to the Exam Prep Bot!\n{status}\n\n"
        "Please upload your *syllabus PDF* to get started.",
        parse_mode="Markdown",
    )


async def cmd_skip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user.id

    if user_state.get(user) != "paper":
        await update.message.reply_text("Please send /start and upload your syllabus PDF first.")
        return

    syllabus = context.user_data.get("syllabus", "")
    if not syllabus:
        await update.message.reply_text("Syllabus not found. Please send /start and upload it again.")
        return

    if not is_paid(user) and get_remaining(user) <= 0:
        await show_paywall(update, user)
        return

    await update.message.reply_text("⏳ Analysing syllabus and predicting high-probability questions…")

    try:
        prompt = (
            "Analyse the following syllabus and produce a high-accuracy exam prediction report.\n\n"
            f"Syllabus:\n{syllabus[:MAX_PDF_CHARS]}\n\n"
            "Your task:\n"
            "1. Identify the TOP 6 most important topics based on depth, breadth, and typical exam weightage.\n"
            "2. For each topic, briefly explain WHY it is likely to be tested.\n"
            "3. Generate 10 high-probability predicted exam questions with confidence: High / Medium.\n\n"
            "Format:\n\n"
            "IMPORTANT TOPICS:\n"
            "1. [Topic] — [Why it's likely to be tested]\n\n"
            "PREDICTED EXAM QUESTIONS:\n"
            "1. [Question] (Confidence: High/Medium)"
        )
        result = await ask_groq(prompt)
        increment_usage(user)
        user_state[user] = "done"

        remaining = get_remaining(user)
        footer = (
            "\n\n✅ Unlimited access – generate anytime!"
            if is_paid(user)
            else f"\n\n🆓 {remaining} free generation(s) remaining."
        )
        await send_long(update, result + footer)

    except Exception as exc:
        logger.error(f"Groq error (skip) user={user}: {exc}")
        await update.message.reply_text("❌ Something went wrong. Please try again with /start.")


async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user.id

    if user_state.get(user) not in ("syllabus", "paper"):
        await update.message.reply_text("Please send /start first.")
        return

    await update.message.reply_text("⏳ Processing your PDF…")

    # Download and extract text
    try:
        tg_file = await update.message.document.get_file()
        local_path = f"file_{user}.pdf"
        await tg_file.download_to_drive(local_path)

        text = ""
        with pdfplumber.open(local_path) as pdf:
            for page in pdf.pages:
                chunk = page.extract_text()
                if chunk:
                    text += chunk
    except Exception as exc:
        logger.error(f"PDF error user={user}: {exc}")
        await update.message.reply_text("❌ Could not read that PDF. Please send a valid file and try again.")
        return

    # ── Syllabus upload ──
    if user_state.get(user) == "syllabus":
        context.user_data["syllabus"] = text
        user_state[user] = "paper"
        await update.message.reply_text(
            "✅ Syllabus received!\n\n"
            "What would you like to do next?\n"
            "📄 Upload a past year paper for trend-based predictions.\n"
            "⏭ Or type /skip to generate questions from the syllabus alone."
        )
        return

    # ── Past paper upload ──
    if user_state.get(user) == "paper":
        if not is_paid(user) and get_remaining(user) <= 0:
            await show_paywall(update, user)
            return

        syllabus = context.user_data.get("syllabus", "")
        await update.message.reply_text("⏳ Analysing past paper patterns and generating predictions…")

        try:
            prompt = (
                "Analyse the syllabus and past year exam paper below and produce a high-accuracy prediction report.\n\n"
                f"Syllabus:\n{syllabus[:MAX_PDF_CHARS]}\n\n"
                f"Past Year Exam Paper:\n{text[:MAX_PDF_CHARS]}\n\n"
                "Your task:\n"
                "1. Cross-reference past paper topics with the syllabus.\n"
                "2. Detect patterns: repeated topics, mark-heavy sections, recurring question styles.\n"
                "3. List the TOP 6 high-probability topics with reasons.\n"
                "4. Generate 10 predicted questions with confidence (High/Medium) and topic name.\n\n"
                "Format:\n\n"
                "PATTERN ANALYSIS:\n"
                "- [Key observation]\n\n"
                "HIGH-PROBABILITY TOPICS:\n"
                "1. [Topic] — [Reason it's likely to repeat]\n\n"
                "PREDICTED EXAM QUESTIONS:\n"
                "1. [Question] (Confidence: High/Medium | Topic: ...)"
            )
            result = await ask_groq(prompt)
            increment_usage(user)
            user_state[user] = "done"

            remaining = get_remaining(user)
            footer = (
                "\n\n✅ Unlimited access – generate anytime!"
                if is_paid(user)
                else f"\n\n🆓 {remaining} free generation(s) remaining."
            )
            await send_long(update, result + footer)

        except Exception as exc:
            logger.error(f"Groq error (pdf) user={user}: {exc}")
            await update.message.reply_text("❌ Something went wrong. Please try again with /start.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Unhandled bot error: {context.error}")

# ─────────────────────────────────────────────
#  Razorpay webhook (Flask)
# ─────────────────────────────────────────────
flask_app = Flask(__name__)


@flask_app.route("/webhook", methods=["POST"])
def razorpay_webhook():
    data = flask_request.json or {}

    if data.get("event") == "payment.captured":
        try:
            notes   = data["payload"]["payment"]["entity"]["notes"]
            user_id = int(notes["telegram_id"])

            with _lock:
                paid_users.add(user_id)
            save_data()

            # Notify user — plain HTTP call, fully thread-safe, no asyncio required
            notify_user_via_api(
                user_id,
                "✅ Payment successful! You now have unlimited access. 🎉\n\nSend /start to continue.",
            )
            logger.info(f"Payment confirmed for user {user_id}")

        except Exception as exc:
            logger.error(f"Webhook processing error: {exc}")

    return "OK", 200


def run_flask() -> None:
    logger.info(f"Webhook server starting on port {WEBHOOK_PORT}")
    flask_app.run(host="0.0.0.0", port=WEBHOOK_PORT, use_reloader=False)

# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("skip",  cmd_skip))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    app.add_error_handler(error_handler)

    logger.info("Bot starting…")
    app.run_polling()
