import os
import socket
import logging
import threading
from datetime import datetime, timedelta
from emoji import demojize
import requests
import pandas as pd
import csv
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass 

# Twitch application credentials
client_id = os.getenv("TWITCH_CLIENT_ID", "XXXXXXXXX")
client_secret = os.getenv("TWITCH_CLIENT_SECRET", "XXXXXXXXX")

# IRC / chat credentials
nickname = os.getenv("TWITCH_NICK", "XXXXXXXXX")
chat_token = os.getenv("TWITCH_CHAT_TOKEN", "XXXXXXXXX")  

# Excel file containing one streamer display-name per row (first column)
streamer_file = os.getenv(
    "STREAMER_CSV", 
    "~/Desktop/NLP_Submission/Data/all_streamer.csv", 
)

# Where to store CSV chat logs
output_folder = os.path.join(os.path.expanduser("~"), "/Desktop/NLP_Submission/Data/1_scraped_data", "scraped_twitch") 
os.makedirs(output_folder, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# OAuth token handling (app access token)
token_url = "https://id.twitch.tv/oauth2/token"
access_token: str | None = None
expires_at: datetime = datetime.min


def get_access_token() -> None:
    global access_token, expires_at

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
    }
    r = requests.post(token_url, data=payload, timeout=15)
    r.raise_for_status()
    data = r.json()
    access_token = data["access_token"]
    expires_at = datetime.utcnow() + timedelta(seconds=data["expires_in"])
    logging.info("Obtained new access token (valid until %s UTC)", expires_at.isoformat(" ", "seconds"))


def ensure_token() -> None:
    if access_token is None or datetime.utcnow() >= expires_at - timedelta(minutes=5):
        get_access_token()


def twitch_headers() -> dict[str, str]:
    ensure_token()
    return {
        "Client-ID": client_id,
        "Authorization": f"Bearer {access_token}",
    }


login_cache: dict[str, tuple[str, str]] = {}  # {display_name: (login, user_id)}

def resolve_login_and_id(display_name: str) -> tuple[str | None, str | None]:
    if display_name in login_cache:
        return login_cache[display_name]

    url = "https://api.twitch.tv/helix/search/channels"
    params = {"query": display_name, "first": 1}
    r = requests.get(url, headers=twitch_headers(), params=params, timeout=15)
    if r.status_code != 200:
        logging.error("Channel search failed for %s: %s", display_name, r.text)
        return None, None

    data = r.json().get("data", [])
    if not data:
        return None, None

    chan = next((c for c in data if c["display_name"] == display_name), data[0])
    login_cache[display_name] = (chan["broadcaster_login"], chan["id"])
    return login_cache[display_name]


# Stream status
def is_streamer_online(display_name: str) -> bool:
    """Return True if the streamer is currently live."""
    _, user_id = resolve_login_and_id(display_name)
    if not user_id:
        return False

    url = "https://api.twitch.tv/helix/streams"
    params = {"user_id": user_id}
    r = requests.get(url, headers=twitch_headers(), params=params, timeout=15)
    if r.status_code != 200:
        logging.error("Stream status check failed for %s: %s", display_name, r.text)
        return False

    return bool(r.json().get("data"))


# Chat capture
def save_to_csv(streamer: str, batch: list[tuple], session_id: int, part: int) -> None:
    if not batch:
        return

    df = pd.DataFrame(
        batch,
        columns=["streamer_name", "session_id", "message", "timestamp"],
    )
    path = os.path.join(output_folder, f"{streamer}_session{session_id}_part{part}.csv")
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    logging.info("Saved %d messages ➜ %s", len(batch), path)


IRC_SERVER = "irc.chat.twitch.tv"
IRC_PORT = 6667


def manage_streamer(display_name: str) -> None:
    login, _ = resolve_login_and_id(display_name)
    if not login:
        logging.warning("%s: no channel found", display_name)
        return

    channel = f"#{login.lower()}"

    while True:
        if not is_streamer_online(display_name):
            logging.info("%s offline. Checking again in 60 s", display_name)
            time.sleep(60)
            continue

        session_id = int(datetime.utcnow().timestamp())
        batch: list[tuple] = []
        part = 1

        with socket.socket() as sock:
            try:
                sock.settimeout(240)
                sock.connect((IRC_SERVER, IRC_PORT))
                sock.sendall(f"PASS oauth:{chat_token}\r\n".encode())
                sock.sendall(f"NICK {nickname}\r\n".encode())
                sock.sendall(f"JOIN {channel}\r\n".encode())
                logging.info("[%s] Connected to %s", display_name, channel)

                while True:
                    raw = sock.recv(2048).decode(errors="ignore")
                    if not raw:
                        break  

                    if raw.startswith("PING"):
                        sock.sendall("PONG :tmi.twitch.tv\r\n".encode())
                        continue

                    batch.append((display_name, session_id, demojize(raw), datetime.utcnow()))
                    if len(batch) >= 500:
                        save_to_csv(display_name, batch, session_id, part)
                        batch.clear()
                        part += 1

            except socket.timeout:
                logging.error("[%s] IRC timeout", display_name)
            except Exception as exc:
                logging.exception("[%s] Error: %s", display_name, exc)
            finally:
                save_to_csv(display_name, batch, session_id, part)
                time.sleep(10)  

def get_streamers_from_csv(path: str) -> list[str]:
    df = pd.read_csv(path)
    return df.iloc[:, 0].dropna().astype(str).tolist()

def main() -> None:
    get_access_token()
    streamers = get_streamers_from_csv(streamer_file)

    for name in streamers:
        threading.Thread(target=manage_streamer, args=(name,), daemon=True).start()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logging.info("Shutdown requested by user. Bye!")


if __name__ == "__main__":
    main()