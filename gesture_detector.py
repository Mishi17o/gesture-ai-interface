import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def init_hands():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )


def detect_gesture(frame, hands):
    """
    Tuned for higher accuracy: stricter open palm, looser others, debug messages.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if not results.multi_hand_landmarks:
        cv2.putText(frame, "No hand", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return None

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        lm = hand_landmarks.landmark

        # Hand size filter (reject tiny/false detections)
        wrist = lm[0]
        middle_tip = lm[12]
        hand_size = math.hypot(wrist.x - middle_tip.x, wrist.y - middle_tip.y)
        if hand_size < 0.18:
            cv2.putText(frame, "Hand too small", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return None

        # Palm facing filter (wrist to middle finger depth small)
        if abs(lm[0].z - lm[9].z) > 0.15:
            cv2.putText(frame, "Palm not facing", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return None

        # Finger curl score (0 = straight, 1 = curled)
        def curl_score(tip, dip, pip, mcp):
            d_tip_mcp = math.hypot(lm[tip].x - lm[mcp].x, lm[tip].y - lm[mcp].y)
            d_dip_mcp = math.hypot(lm[dip].x - lm[mcp].x, lm[dip].y - lm[mcp].y)
            score = (d_tip_mcp - d_dip_mcp) / d_tip_mcp if d_tip_mcp > 0 else 0
            return max(0, min(1, score))

        # ── Gesture checks – ORDER: specific first, general last ────────────────

        # 1. Pinch / OK – thumb + index very close, other fingers extended
        pinch_dist = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        others_extended = all(curl_score(t,d,p,m) < 0.22 for t,d,p,m in [(12,11,10,9), (16,15,14,13), (20,19,18,17)])
        if pinch_dist < 0.032 and others_extended:
            return "pinch"

        # 2. Pointing – only index extended, others curled
        index_extended = curl_score(8,7,6,5) < 0.18 and lm[8].y < lm[6].y - 0.025
        others_curled = all(curl_score(t,d,p,m) > 0.38 for t,d,p,m in [(12,11,10,9), (16,15,14,13), (20,19,18,17)])
        if index_extended and others_curled:
            return "pointing"

        # 3. Thumbs up – thumb clearly up + fingers curled
        thumb_up = lm[4].y < lm[2].y - 0.06 and lm[4].x > lm[3].x + 0.01
        fingers_curled = all(curl_score(t,d,p,m) > 0.38 for t,d,p,m in [(8,7,6,5), (12,11,10,9), (16,15,14,13), (20,19,18,17)])
        if thumb_up and fingers_curled:
            return "thumbs_up"

        # 4. Thumbs down – thumb down + fingers curled
        thumb_down = lm[4].y > lm[2].y + 0.06
        if thumb_down and fingers_curled:
            return "thumbs_down"

        # 5. Open palm – last & strict: all fingers very extended + wide spread
        fingers_extended = all(curl_score(t,d,p,m) < 0.18 for t,d,p,m in [(8,7,6,5), (12,11,10,9), (16,15,14,13), (20,19,18,17)])
        spread = math.hypot(lm[8].x - lm[20].x, lm[8].y - lm[20].y) > 0.22  # stricter spread
        if fingers_extended and spread:
            return "open_palm"

        # Fallback
        cv2.putText(frame, "Unknown / partial gesture", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        return None

    return None