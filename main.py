import cv2
import time
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import ollama
from speech import init_tts, speak

# ── SETTINGS ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
COOLDOWN_SECONDS = 3.0
CONFIDENCE_THRESHOLD = 0.55

GESTURE_RESPONSES = {
    "Thumb_Up":     "Yes! Great choice.",
    "Thumb_Down":   "No, let's try again.",
    "Pointing_Up":  "Pointing detected.",
    "Open_Palm":    "Open palm – stopping.",
    "Victory":      "Victory / peace sign!",
    "ILoveYou":     "I love you sign!",
    "Closed_Fist":  "Fist / stop.",
}


def speak_in_thread(engine, text):
    threading.Thread(target=speak, args=(engine, text), daemon=True).start()


def get_ollama_response(gesture_name):
    prompt = f"""
    You are a friendly AI in a gesture-controlled interface.
    User did '{gesture_name}' gesture.
    Give a short, fun, positive reply (max 12 words).
    """
    try:
        response = ollama.generate(model='phi3:mini', prompt=prompt)  # or llama3.1:8b
        return response['response'].strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return f"I saw {gesture_name}. Cool!"


def main():
    print("=== Gesture AI + Ollama ===")
    print(f"Camera index: {CAMERA_INDEX}")
    print("Press 'q' to quit\n")

    # Load gesture model (no invalid args)
    base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)   # lower res = faster
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not cap.isOpened():
        print("Camera failed")
        return

    tts_engine = init_tts()

    last_gesture = None
    last_trigger_time = 0.0

    window_name = "Gesture AI + Ollama – press q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    print("Starting...\n")
    timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame failed")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        recognition_result = recognizer.recognize_for_video(mp_image, timestamp)
        timestamp += 1

        gesture_name = "None"
        confidence = 0.0

        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
            gesture_name = top_gesture.category_name
            confidence = top_gesture.score

            display_text = f"{gesture_name} ({confidence:.2f})"
            color = (0, 255, 130) if confidence > 0.7 else (0, 165, 255)
            cv2.putText(frame, display_text, (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            if confidence > CONFIDENCE_THRESHOLD and (gesture_name != last_gesture or time.time() - last_trigger_time > COOLDOWN_SECONDS):
                response = get_ollama_response(gesture_name)
                print(f"Ollama: {response}")
                speak_in_thread(tts_engine, response)
                last_gesture = gesture_name
                last_trigger_time = time.time()

        else:
            cv2.putText(frame, "No gesture", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()