import logging
import os
import whisper
from pydub import AudioSegment
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# --- Configuration ---
# Get the token from the environment variable set in the hosting service
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("No TELEGRAM_TOKEN environment variable set!")

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load the Whisper model
try:
    logger.info("Loading Whisper model...")
    model = whisper.load_model("base") # "base" is a good balance of speed and accuracy
    logger.info("Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")
    raise

# --- Bot Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! Send me a voice message, and I'll transcribe it for you.")

async def transcribe_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.voice:
        return

    processing_message = await update.message.reply_text("Downloading and transcribing...")

    try:
        voice_file_id = update.message.voice.file_id
        file = await context.bot.get_file(voice_file_id)

        # Define file paths
        oga_filename = f"{voice_file_id}.oga"
        mp3_filename = f"{voice_file_id}.mp3"
        
        await file.download_to_drive(oga_filename)

        # Convert OGA to MP3
        audio = AudioSegment.from_ogg(oga_filename)
        audio.export(mp3_filename, format="mp3")

        # Transcribe
        result = model.transcribe(mp3_filename)
        transcribed_text = result["text"]
        
        if not transcribed_text.strip():
            reply_text = "I couldn't detect any speech in the audio."
        else:
            reply_text = f"📝 **Transcription:**\n\n_{transcribed_text}_"

        await processing_message.edit_text(reply_text, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        await processing_message.edit_text("Sorry, an error occurred while processing your message.")
    
    finally:
        # Clean up temporary files
        if os.path.exists(oga_filename):
            os.remove(oga_filename)
        if os.path.exists(mp3_filename):
            os.remove(mp3_filename)

# --- Main Bot Setup ---
def main():
    logger.info("Starting bot...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE, transcribe_voice_message))
    
    logger.info("Bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()
