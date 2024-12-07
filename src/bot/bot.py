import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, FSInputFile, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
import logging
from common.utils import load_config, save_json

logging.basicConfig(level=logging.INFO)

config = load_config("config/config.json")

TOKEN = config["telegram"]["token"]  # Замените на токен вашего бота

bot = Bot(token=TOKEN)
dp = Dispatcher()

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())