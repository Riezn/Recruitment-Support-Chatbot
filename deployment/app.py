
from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters


updater = Updater("your_own_API_Token got from BotFather",
                  use_context=True)
  
  
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Halo, selamat datang. Ada yang bisa dibantu?")


def help(update: Update, context: CallbackContext):
    update.message.reply_text("Hubungi developer untuk bantuan lebih lanjut pada https://github.com/Riezn/.")


def unknown(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Maaf, Kak. Aku tidak mengerti perintah '%s'." % update.message.text)