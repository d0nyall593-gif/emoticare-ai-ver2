import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EmotiCare AI",
    page_icon="💙",
    layout="centered"
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "stage": "welcome",
    "first_emotion": None,
    "first_confidence": 0.0,
    "second_emotion": None,
    "second_confidence": 0.0,
    "actual_emotion": None,
    "messages": [],
    "conversation_count": 0,
    "final_feeling": None,
    "user_input": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# EMOTION MODEL
# ============================================================

@st.cache_resource
def load_emotion_model():

    from transformers import pipeline

    return pipeline(
        "image-classification",
        model="trpakov/vit-face-expression"
    )


# ============================================================
# EMOTION EMOJIS
# ============================================================

EMOTION_EMOJIS = {
    "happy": "😊",
    "sad": "😔",
    "angry": "😡",
    "fear": "😨",
    "surprise": "😮",
    "disgust": "🤢",
    "neutral": "😐",
}


# ============================================================
# EMOTION ANALYSIS
# ============================================================

def analyze_emotion(photo, emotion_model):

    image = Image.open(photo).convert("RGB")

    results = emotion_model(image)

    if not results:
        return None, 0.0

    best = results[0]

    emotion = best["label"].lower()
    confidence = float(best["score"]) * 100

    return emotion, confidence


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    return text.strip()


# ============================================================
# KEYWORD DETECTION
# ============================================================

def contains_any(text, words):

    text = text.lower()

    for word in words:

        if word in text:
            return True

    return False


# ============================================================
# CONVERSATION ENGINE
# ============================================================

def generate_response(user_text, emotion, history):

    text = clean_text(user_text)
    lower = text.lower()

    # --------------------------------------------------------
    # SERIOUS DISTRESS
    # --------------------------------------------------------

    serious_words = [
        "kill myself",
        "suicide",
        "want to die",
        "hurt myself",
        "self harm",
        "self-harm",
    ]

    if contains_any(lower, serious_words):

        return (
            "I'm really sorry you're dealing with something "
            "this heavy. Please don't handle it alone. "
            "Consider talking to a parent, teacher, counselor, "
            "or another trusted person who can support you."
        )

    # --------------------------------------------------------
    # SCHOOL
    # --------------------------------------------------------

    if contains_any(
        lower,
        [
            "school",
            "exam",
            "test",
            "homework",
            "teacher",
            "class",
            "grade",
            "marks",
            "assignment",
        ],
    ):

        return (
            "School pressure can definitely affect how we feel. "
            "What happened today that bothered you the most?"
        )

    # --------------------------------------------------------
    # FRIENDS
    # --------------------------------------------------------

    if contains_any(
        lower,
        [
            "friend",
            "friends",
            "best friend",
            "ignored me",
            "ignore me",
            "left me",
            "excluded me",
            "group",
        ],
    ):

        return (
            "That sounds difficult, especially when friends are "
            "involved. Do you think they knew how their actions "
            "made you feel?"
        )

    # --------------------------------------------------------
    # FAMILY
    # --------------------------------------------------------

    if contains_any(
        lower,
        [
            "mom",
            "mum",
            "mother",
            "dad",
            "father",
            "parent",
            "parents",
            "brother",
            "sister",
            "family",
        ],
    ):

        return (
            "Family situations can be complicated. 💙 "
            "Would you like to tell me a little more about "
            "what happened?"
        )

    # --------------------------------------------------------
    # TIRED
    # --------------------------------------------------------

    if contains_any(
        lower,
        [
            "tired",
            "sleep",
            "sleepy",
            "exhausted",
            "no energy",
            "energy",
        ],
    ):

        return (
            "It sounds like you might really need some time "
            "to recharge. Have you been getting enough rest "
            "lately?"
        )

    # --------------------------------------------------------
    # ANGRY
    # --------------------------------------------------------

    if contains_any(
        lower,
        [
            "angry",
            "mad",
            "annoyed",
            "annoying",
            "furious",
            "rage",
        ],
    ):

        return (
            "It sounds like something really got under your skin. "
            "What was the part of the situation that made you "
            "the most angry?"
        )

    # --------------------------------------------------------
    # SAD
    # --------------------------------------------------------

    if contains_any(
        lower,
        [
            "sad",
            "lonely",
            "alone",
            "cry",
            "crying",
            "upset",
            "hurt",
            "heartbroken",
        ],
    ):

        return (
            "I'm sorry you're going through that. 💙 "
            "What do you think is making this feeling especially "
            "strong right now?"
        )

    # --------------------------------------------------------
    # WORRY / STRESS
    # --------------------------------------------------------

    if contains_any(
        lower,
        [
            "stress",
            "stressed",
            "worried",
            "worry",
            "anxious",
            "anxiety",
            "nervous",
            "pressure",
        ],
    ):

        return (
            "That sounds like a lot to carry. "
            "Is there one particular thing you're worrying "
            "about the most?"
        )

    # --------------------------------------------------------
    # HAPPY
    # --------------------------------------------------------

    if contains_any(
        lower,
        [
            "happy",
            "great",
            "amazing",
            "good",
            "excited",
            "fun",
            "awesome",
            "wonderful",
        ],
    ):

        return (
            "That's nice to hear! 😊 "
            "What happened that made your day better?"
        )

    # --------------------------------------------------------
    # POSITIVE CHANGE
    # --------------------------------------------------------

    if contains_any(
        lower,
        [
            "better now",
            "feel better",
            "feeling better",
            "okay now",
            "fine now",
            "i'm okay",
            "im okay",
            "i am okay",
        ],
    ):

        return (
            "I'm glad things feel a little better. 💙 "
            "What do you think helped you feel that way?"
        )

    # --------------------------------------------------------
    # FIRST FOLLOW-UP
    # --------------------------------------------------------

    if len(history) <= 2:

        return (
            "Thanks for telling me. 💙 "
            "Can you tell me a little more about what happened?"
        )

    # --------------------------------------------------------
    # SECOND FOLLOW-UP
    # --------------------------------------------------------

    if len(history) <= 4:

        return (
            "I understand. Sometimes talking about what happened "
            "can make things a little clearer. "
            "What do you think you need right now?"
        )

    # --------------------------------------------------------
    # LATER FOLLOW-UPS
    # --------------------------------------------------------

    responses = [

        (
            "That makes sense. 💙 "
            "If you could change one thing about what happened, "
            "what would it be?"
        ),

        (
            "Thanks for being honest with me. "
            "Do you feel like this is something you can work "
            "through yourself, or would talking to someone "
            "you trust help?"
        ),

        (
            "I hear you. "
            "What do you think would make the situation feel "
            "a little easier right now?"
        ),

        (
            "That's understandable. "
            "Has this been bothering you for a while, "
            "or did it happen recently?"
        ),
    ]

    index = (
        st.session_state.conversation_count
        % len(responses)
    )

    return responses[index]


# ============================================================
# RESET
# ============================================================

def reset_app():

    for key in DEFAULTS:

        if key in st.session_state:

            del st.session_state[key]

    st.experimental_rerun()


# ============================================================
# WELCOME
# ============================================================

if st.session_state.stage == "welcome":

    st.title("💙 EmotiCare AI")

    st.subheader(
        "A little space to check in with yourself."
    )

    st.write(
        """
        EmotiCare uses AI to estimate your visible facial
        expression and guides you through a short conversation
        about how you're feeling.
        """
    )

    st.info(
        """
        📸 Your camera is only used when you choose to
        take an emotion-check photo.
        """
    )

    st.warning(
        """
        ⚠️ The camera cannot know exactly how you feel.

        It only estimates the facial expression visible
        in the photo. Your own explanation is more important.
        """
    )

    if st.button(
        "💙 Start Check-in",
        use_container_width=True,
        type="primary",
    ):

        st.session_state.stage = "first_scan"

        st.experimental_rerun()


# ============================================================
# FIRST SCAN
# ============================================================

elif st.session_state.stage == "first_scan":

    st.title("📸 First Emotion Check")

    st.write(
        "Take a photo so the AI can estimate your expression."
    )

    photo = st.camera_input(
        "Take your first photo"
    )

    if photo:

        with st.spinner(
            "Loading the emotion AI..."
        ):

            try:

                emotion_model = load_emotion_model()

            except Exception as error:

                st.error(
                    "The emotion AI could not load."
                )

                st.exception(error)

                st.stop()

        with st.spinner(
            "Analyzing your expression..."
        ):

            try:

                emotion, confidence = analyze_emotion(
                    photo,
                    emotion_model,
                )

                if emotion is None:

                    st.error(
                        "I couldn't detect an expression."
                    )

                else:

                    st.session_state.first_emotion = emotion
                    st.session_state.first_confidence = confidence

                    emoji = EMOTION_EMOJIS.get(
                        emotion,
                        "🙂",
                    )

                    st.success(
                        f"{emoji} Possible expression: "
                        f"**{emotion.capitalize()}**"
                    )

                    st.caption(
                        f"AI estimate confidence: "
                        f"{confidence:.1f}%"
                    )

                    st.write(
                        f"Does **{emotion}** match how you "
                        "actually feel?"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        if st.button(
                            "✅ Yes",
                            use_container_width=True,
                        ):

                            st.session_state.actual_emotion = emotion

                            st.session_state.stage = (
                                "conversation"
                            )

                            st.experimental_rerun()

                    with col2:

                        if st.button(
                            "❌ No",
                            use_container_width=True,
                        ):

                            st.session_state.stage = (
                                "correct_emotion"
                            )

                            st.experimental_rerun()

            except Exception as error:

                st.error(
                    "The emotion AI could not analyze "
                    "the photo."
                )

                st.exception(error)


# ============================================================
# CORRECT EMOTION
# ============================================================

elif st.session_state.stage == "correct_emotion":

    st.title("💭 You know yourself best")

    st.write(
        """
        The camera can make mistakes.

        Tell EmotiCare how you're actually feeling.
        """
    )

    choices = [
        "Happy",
        "Sad",
        "Angry",
        "Worried",
        "Stressed",
        "Tired",
        "Excited",
        "Confused",
        "Neutral",
    ]

    selected = st.selectbox(
        "How are you feeling?",
        choices,
    )

    if st.button(
        "Continue 💙",
        use_container_width=True,
        type="primary",
    ):

        st.session_state.actual_emotion = selected.lower()

        st.session_state.stage = "conversation"

        st.experimental_rerun()


# ============================================================
# CONVERSATION
# ============================================================

elif st.session_state.stage == "conversation":

    emotion = st.session_state.actual_emotion

    emoji = EMOTION_EMOJIS.get(
        emotion,
        "💙",
    )

    st.title("💬 Talk to EmotiCare")

    st.caption(
        f"{emoji} We're checking in about how you're feeling."
    )

    # --------------------------------------------------------
    # INITIAL MESSAGE
    # --------------------------------------------------------

    if len(st.session_state.messages) == 0:

        opening = (
            f"I noticed a possible {emotion} expression. "
            "Of course, the camera can be wrong, so I want "
            "to hear it from you. Why do you think you're "
            f"feeling {emotion}?"
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": opening,
            }
        )

    # --------------------------------------------------------
    # DISPLAY CONVERSATION
    # Compatible with Streamlit 1.22
    # --------------------------------------------------------

    for message in st.session_state.messages:

        if message["role"] == "assistant":

            st.markdown(
                "🤖 **EmotiCare:** "
                + message["content"]
            )

        else:

            st.markdown(
                "👤 **You:** "
                + message["content"]
            )

    st.divider()

    # --------------------------------------------------------
    # TEXT INPUT
    # --------------------------------------------------------

    user_message = st.text_input(
        "Tell me what's going on...",
        key="user_input",
    )

    if st.button(
        "Send 💬",
        use_container_width=True,
    ):

        user_message = clean_text(
            user_message
        )

        if user_message:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_message,
                }
            )

            st.session_state.conversation_count += 1

            response = generate_response(
                user_message,
                emotion,
                st.session_state.messages,
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

            st.session_state.user_input = ""

            st.experimental_rerun()

        else:

            st.warning(
                "Write something first 🙂"
            )

    # --------------------------------------------------------
    # END CONVERSATION
    # --------------------------------------------------------

    st.divider()

    st.write(
        "When you're ready, let's check in one more time."
    )

    if st.button(
        "💙 How do you feel now?",
        use_container_width=True,
        type="primary",
    ):

        st.session_state.stage = "second_scan"

        st.experimental_rerun()


# ============================================================
# SECOND SCAN
# ============================================================

elif st.session_state.stage == "second_scan":

    st.title("💙 How do you feel now?")

    st.write(
        """
        Take another photo.

        EmotiCare will make a second estimate so we can
        compare the visible expression with your first scan.
        """
    )

    st.caption(
        "Remember: this is only an AI estimate."
    )

    photo = st.camera_input(
        "Take your final photo"
    )

    if photo:

        with st.spinner(
            "Analyzing your expression again..."
        ):

            try:

                emotion_model = load_emotion_model()

                emotion, confidence = analyze_emotion(
                    photo,
                    emotion_model,
                )

                if emotion is None:

                    st.error(
                        "I couldn't analyze the second photo."
                    )

                else:

                    st.session_state.second_emotion = emotion

                    st.session_state.second_confidence = (
                        confidence
                    )

                    st.session_state.stage = "result"

                    st.experimental_rerun()

            except Exception as error:

                st.error(
                    "The second emotion scan failed."
                )

                st.exception(error)


# ============================================================
# RESULTS
# ============================================================

elif st.session_state.stage == "result":

    first = st.session_state.first_emotion
    second = st.session_state.second_emotion

    st.title("📊 Your Check-in")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Before")

        st.write(
            f"{EMOTION_EMOJIS.get(first, '🙂')} "
            f"**{first.capitalize()}**"
        )

        st.caption(
            f"AI estimate: "
            f"{st.session_state.first_confidence:.1f}%"
        )

    with col2:

        st.subheader("Now")

        st.write(
            f"{EMOTION_EMOJIS.get(second, '🙂')} "
            f"**{second.capitalize()}**"
        )

        st.caption(
            f"AI estimate: "
            f"{st.session_state.second_confidence:.1f}%"
        )

    st.divider()

    if first == second:

        st.info(
            f"""
            The AI estimated **{second}** in both photos.

            This does not necessarily mean you feel the same,
            because facial-expression AI can be inaccurate.
            """
        )

    else:

        st.success(
            f"""
            The AI's visible-expression estimate changed
            from **{first}** to **{second}**.
            """
        )

    # --------------------------------------------------------
    # USER'S OWN FEELING
    # --------------------------------------------------------

    st.subheader(
        "But how do YOU feel now?"
    )

    final_feeling = st.radio(
        "Choose the answer that feels closest:",
        [
            "😊 I feel better",
            "🙂 I feel a little better",
            "😐 I feel about the same",
            "😔 I still don't feel good",
        ],
    )

    if st.button(
        "Finish 💙",
        use_container_width=True,
        type="primary",
    ):

        st.session_state.final_feeling = final_feeling

        st.session_state.stage = "goodbye"

        st.experimental_rerun()


# ============================================================
# GOODBYE
# ============================================================

elif st.session_state.stage == "goodbye":

    st.title("💙 Thank you for checking in")

    feeling = st.session_state.final_feeling

    if "I feel better" in feeling:

        st.success(
            """
            I'm really glad you're feeling better. 💙

            Remember to be kind to yourself.
            """
        )

    elif "little better" in feeling:

        st.success(
            """
            Even a little improvement matters. 💙

            Take things one step at a time.
            """
        )

    elif "about the same" in feeling:

        st.info(
            """
            That's completely okay. 💙

            You don't have to feel better immediately.
            Give yourself some time.
            """
        )

    else:

        st.info(
            """
            I'm sorry you're still having a difficult moment. 💙

            Consider talking to someone you trust if you
            need some extra support.
            """
        )

    st.write(
        "Thank you for spending a moment with EmotiCare."
    )

    st.divider()

    if st.button(
        "🔄 Start Again",
        use_container_width=True,
    ):

        reset_app()
