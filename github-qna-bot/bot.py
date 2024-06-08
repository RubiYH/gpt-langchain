import os
import discord
from main import ask_github

from dotenv import load_dotenv

load_dotenv(dotenv_path='../.env')

bot = discord.Bot()

@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")

@bot.slash_command(name="ask", description="Ask a question!")
async def ask(ctx, question: discord.Option(str)):
    await ctx.defer()

    get_response = ask_github(question)
    await ctx.respond(f"{ctx.author.mention} Here is the answer: \n{get_response}")
    print("\n\nDone!")

bot.run(os.environ['DISCORD_BOT_API_KEY'])