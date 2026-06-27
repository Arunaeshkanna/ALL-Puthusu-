from youtube_transcript_api import YouTubeTranscriptApi


def extract_youtube_id(url_or_id: str) -> str:
    value = url_or_id.strip()
    if "watch?v=" in value:
        return value.split("watch?v=", 1)[1].split("&", 1)[0]
    if "youtu.be/" in value:
        return value.split("youtu.be/", 1)[1].split("?", 1)[0]
    return value


def load_youtube_transcript(url_or_id: str) -> str:
    video_id = extract_youtube_id(url_or_id)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join(item["text"] for item in transcript)

