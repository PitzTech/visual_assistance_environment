let lastVoiceOutput = null
let isPlaying = false
let lastPlayTime = 0
const SPEECH_COOLDOWN = 3000 // 3 seconds in milliseconds

export const generateAndPlaySpeech = async (text) => {
  if (isPlaying) return
  if (text == lastVoiceOutput) return

  // Check cooldown time
  const currentTime = Date.now()
  if (currentTime - lastPlayTime < SPEECH_COOLDOWN) {
    return // Ignore request during cooldown
  }

  lastVoiceOutput = text
  lastPlayTime = currentTime
  isPlaying = true

  try {
    // Make the API request
    const response = await fetch('http://localhost:8880/v1/audio/speech', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        input: text,
        voice: "af_kore",
        response_format: "mp3",
        download_format: "mp3",
        stream: true,
        speed: 1,
        return_download_link: true,
        lang_code: "p"
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    console.log({ response })

    // Convert response to blob
    const audioBlob = await response.blob();

    // Create object URL from blob
    const audioUrl = URL.createObjectURL(audioBlob);

    // Create and play audio without HTML player
    const audio = new Audio(audioUrl);

    // Play the audio
    await audio.play();

    // Optional: Clean up the URL after playing
    audio.addEventListener('ended', () => {
      URL.revokeObjectURL(audioUrl);
      isPlaying = false
    });

    return audio; // Return audio object for further control if needed

  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}
