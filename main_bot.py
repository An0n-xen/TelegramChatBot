from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import os, requests
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

# Loading env files
load_dotenv(find_dotenv(), override=True)


# Token for test bot (https://t.me/Supersede_test_bot)
token = os.environ.get("TELEGRAM_BOT_TESTING_KEY")

# Token for bot submitted by Snv Dev (https://t.me/Boxpark_CustomerServiceBot)
token2 = os.environ.get("TELEGRAM_BOT_MAIN_KEY")
url = "http://127.0.0.1:8000/"


class sendMessage(BaseModel):
    update_id: int
    message: str


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hello, welcome to Supersede. What can I do for you?"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("What do you need?")


async def handle_text_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    user_message = update.message.text
    print(f"User: {update.message.from_user.first_name} -> {user_message}")
    response = processInput(user_message)
    await update.message.reply_text(response)


def processInput(user_message: str) -> str:
    processed_user_json: sendMessage = {"update_id": 0, "message": user_message}
    bot_response = requests.post(url, json=processed_user_json)
    return bot_response.json()["message"]


def main() -> None:
    print("Starting bot")

    application = Application.builder().token(token).build()

    # Adding commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message)
    )

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
