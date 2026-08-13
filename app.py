import re
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
    "conversation_topic": None,
    "safety_mode": False,
    "asked_safety_question": False,
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
# TEXT HELPERS
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    return text.strip()


def contains_any(text, phrases):

    text = text.lower()

    for phrase in phrases:
        if phrase in text:
            return True

    return False


def word_count(text):

    return len(re.findall(r"\b\w+\b", text))


# ============================================================
# SITUATION DETECTOR
# ============================================================

def detect_situation(text):

    text = text.lower()

    # --------------------------------------------------------
    # IMMEDIATE DANGER / ACCIDENT / INJURY
    # --------------------------------------------------------

    if contains_any(
        text,
        [
            "got hit by a car",
            "hit by a car",
            "car hit me",
            "car accident",
            "car crash",
            "traffic accident",
            "road accident",
            "motorcycle accident",
            "bike accident",
            "bicycle accident",
            "got injured",
            "i am injured",
            "i'm injured",
            "bleeding",
            "broken bone",
            "broke my arm",
            "broke my leg",
            "can't breathe",
            "cannot breathe",
            "severe pain",
            "badly hurt",
            "seriously hurt",
        ],
    ):
        return "accident_injury"

    # --------------------------------------------------------
    # SELF-HARM / SUICIDE
    # --------------------------------------------------------

    if contains_any(
        text,
        [
            "kill myself",
            "killing myself",
            "suicide",
            "want to die",
            "wish i was dead",
            "wish i were dead",
            "end my life",
            "hurt myself",
            "harm myself",
            "self harm",
            "self-harm",
        ],
    ):
        return "self_harm"

    # --------------------------------------------------------
    # BULLYING
    # --------------------------------------------------------

    if contains_any(
        text,
        [
            "bully",
            "bullied",
            "bullying",
            "being bullied",
            "they keep hitting me",
            "they hit me",
            "people make fun of me",
            "they make fun of me",
            "they laugh at me",
            "they pick on me",
            "picked on",
            "harassing me",
            "harassed me",
        ],
    ):
        return "bullying"

    # --------------------------------------------------------
    # FRIENDSHIP
    # --------------------------------------------------------

    if contains_any(
        text,
        [
            "best friend",
            "my friend",
            "my friends",
            "friend stopped",
            "friends stopped",
            "friend ignored",
            "friends ignored",
            "friend left me",
            "friends left me",
            "friend betrayed",
            "friendship",
            "they don't talk to me",
            "they stopped talking",
        ],
    ):
        return "friendship"

    # --------------------------------------------------------
    # FAMILY
    # --------------------------------------------------------

    if contains_any(
        text,
        [
            "my mom",
            "my mum",
            "my mother",
            "my dad",
            "my father",
            "my parents",
            "my brother",
            "my sister",
            "my family",
            "family problem",
            "family problems",
            "parents fighting",
            "mom and dad fighting",
        ],
    ):
        return "family"

    # --------------------------------------------------------
    # SCHOOL
    # --------------------------------------------------------

    if contains_any(
        text,
        [
            "school",
            "exam",
            "exams",
            "test",
            "tests",
            "homework",
            "assignment",
            "teacher",
            "class",
            "grades",
            "marks",
            "school work",
        ],
    ):
        return "school"

    # --------------------------------------------------------
    # LONELINESS
    # --------------------------------------------------------

    if contains_any(
        text,
        [
            "lonely",
            "alone",
            "nobody cares",
            "no one cares",
            "no friends",
            "have no friends",
            "feel left out",
            "left out",
            "isolated",
        ],
    ):
        return "loneliness"

    # --------------------------------------------------------
    # ANXIETY / STRESS
    # --------------------------------------------------------

    if contains_any(
        text,
        [
            "stressed",
            "stress",
            "anxious",
            "anxiety",
            "worried",
            "worrying",
            "nervous",
            "panic",
            "overwhelmed",
            "too much pressure",
            "pressure",
        ],
    ):
        return "stress"

    # --------------------------------------------------------
    # ANGER
    # --------------------------------------------------------

    if contains_any(
        text,
        [
            "angry",
            "mad",
            "furious",
            "rage",
            "annoyed",
            "annoying me",
            "pissed off",
        ],
    ):
        return "anger"

    # --------------------------------------------------------
    # SADNESS
    # --------------------------------------------------------

    if contains_any(
        text,
        [
            "sad",
            "crying",
            "cry",
            "upset",
            "heartbroken",
            "hurt",
            "feeling down",
            "feel down",
        ],
    ):
        return "sadness"

    # --------------------------------------------------------
    # TIREDNESS
    # --------------------------------------------------------

    if contains_any(
        text,
        [
            "tired",
            "exhausted",
            "sleepy",
            "no energy",
            "can't sleep",
            "cannot sleep",
            "not sleeping",
        ],
    ):
        return "tiredness"

    # --------------------------------------------------------
    # POSITIVE EVENT
    # --------------------------------------------------------

    if contains_any(
        text,
        [
            "won",
            "winning",
            "passed",
            "birthday",
            "got an award",
            "got a prize",
            "promoted",
            "good news",
            "great news",
            "best day",
            "amazing day",
            "really happy",
            "so happy",
        ],
    ):
        return "positive"

    return None


# ============================================================
# SAFETY DETECTOR
# ============================================================

def is_immediate_danger(text):

    text = text.lower()

    return contains_any(
        text,
        [
            "still in danger",
            "someone is attacking me",
            "someone is hurting me",
            "being attacked",
            "attacking me",
            "can't breathe",
            "cannot breathe",
            "bleeding badly",
            "bleeding a lot",
            "unconscious",
        ],
    )


# ============================================================
# SITUATION-AWARE RESPONSE ENGINE
# ============================================================

def generate_response(user_text, emotion, history):

    text = clean_text(user_text)
    lower = text.lower()

    situation = detect_situation(text)

    # --------------------------------------------------------
    # SELF-HARM
    # --------------------------------------------------------

    if situation == "self_harm":

        st.session_state.safety_mode = True

        return (
            "I'm really sorry you're dealing with something "
            "this painful. 💙 I'm glad you told me instead of "
            "keeping it to yourself. Please don't stay alone "
            "with these feelings. Is there a trusted adult, "
            "family member, teacher, counselor, or friend "
            "you can be with right now?"
        )

    # --------------------------------------------------------
    # ACCIDENT / INJURY
    # --------------------------------------------------------

    if situation == "accident_injury":

        st.session_state.safety_mode = True

        if not st.session_state.asked_safety_question:

            st.session_state.asked_safety_question = True

            return (
                "I'm really sorry that happened to you. 💙 "
                "Being hit by a car can be very serious. "
                "Are you safe right now, and are you injured "
                "or still in danger?"
            )

        if is_immediate_danger(text):

            return (
                "Your safety comes first right now. Please "
                "get the attention of a nearby trusted adult "
                "or emergency help immediately. Don't worry "
                "about continuing the conversation with me "
                "until you're safe."
            )

        return (
            "That sounds frightening. I'm glad you're able "
            "to talk about it. If you're injured, please make "
            "sure a trusted adult or medical professional "
            "knows what happened. If you're safe, you can "
            "tell me what happened next."
        )

    # --------------------------------------------------------
    # BULLYING
    # --------------------------------------------------------

    if situation == "bullying":

        return (
            "I'm sorry you're being treated that way. 💙 "
            "You don't deserve to be bullied. If this is "
            "happening at school, telling a teacher, counselor, "
            "parent, or another trusted adult can help. "
            "Do you feel safe around the person who is "
            "bullying you?"
        )

    # --------------------------------------------------------
    # FRIENDSHIP
    # --------------------------------------------------------

    if situation == "friendship":

        return (
            "That can really hurt, especially when it's "
            "someone you care about. 💙 What happened "
            "between you and your friend?"
        )

    # --------------------------------------------------------
    # FAMILY
    # --------------------------------------------------------

    if situation == "family":

        return (
            "Family problems can be really difficult because "
            "you can't always just walk away from them. 💙 "
            "What happened at home?"
        )

    # --------------------------------------------------------
    # SCHOOL
    # --------------------------------------------------------

    if situation == "school":

        return (
            "School pressure can definitely affect how you "
            "feel. 📚 What part of school is bothering you "
            "the most right now?"
        )

    # --------------------------------------------------------
    # LONELINESS
    # --------------------------------------------------------

    if situation == "loneliness":

        return (
            "Feeling alone can be really heavy. 💙 "
            "I'm here to listen. Is it that you don't have "
            "anyone around right now, or that you feel like "
            "the people around you don't understand you?"
        )

    # --------------------------------------------------------
    # STRESS
    # --------------------------------------------------------

    if situation == "stress":

        return (
            "It sounds like you've got a lot on your mind. 💙 "
            "What's the biggest thing causing the pressure "
            "right now?"
        )

    # --------------------------------------------------------
    # ANGER
    # --------------------------------------------------------

    if situation == "anger":

        return (
            "It sounds like something really upset you. "
            "Before we worry about fixing it, what exactly "
            "happened that made you angry?"
        )

    # --------------------------------------------------------
    # SADNESS
    # --------------------------------------------------------

    if situation == "sadness":

        return (
            "I'm sorry you're going through that. 💙 "
            "Do you want to tell me what happened?"
        )

    # --------------------------------------------------------
    # TIREDNESS
    # --------------------------------------------------------

    if situation == "tiredness":

        return (
            "You sound really worn out. 😴 "
            "Has something been keeping you from getting "
            "enough rest, or have you just had a really "
            "busy day?"
        )

    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    if situation == "positive":

        return (
            "That's awesome! 😊 I'd love to hear more. "
            "What happened?"
        )

    # ========================================================
    # CONTEXT FOLLOW-UP
    # ========================================================

    previous_topic = st.session_state.conversation_topic

    if previous_topic == "accident_injury":

        return (
            "I'm glad you're able to talk about it. 💙 "
            "How are you feeling about what happened now?"
        )

    if previous_topic == "bullying":

        return (
            "I understand. You shouldn't have to deal with "
            "that by yourself. Is there someone you trust "
            "who knows what's happening?"
        )

    if previous_topic == "friendship":

        return (
            "I hear you. Friend problems can take a lot out "
            "of you. Do you want things to get better with "
            "your friend, or do you think you need some space?"
        )

    if previous_topic == "family":

        return (
            "I understand. 💙 How has this situation been "
            "affecting you personally?"
        )

    if previous_topic == "school":

        return (
            "That sounds stressful. What would make the "
            "school situation a little easier for you?"
        )

    if previous_topic == "loneliness":

        return (
            "I'm listening. 💙 What would make you feel "
            "less alone right now?"
        )

    if previous_topic == "stress":

        return (
            "Let's take it one step at a time. "
            "Which part of the situation feels hardest "
            "to deal with?"
        )

    # ========================================================
    # EMOTION-SPECIFIC FALLBACK
    # ========================================================

    if emotion == "sad":

        return (
            "I'm listening. 💙 You don't have to explain "
            "everything perfectly. What happened?"
        )

    if emotion == "angry":

        return (
            "It sounds like something really bothered you. "
            "What happened?"
        )

    if emotion == "fear":

        return (
            "You seem to be dealing with something "
            "frightening. What happened?"
        )

    if emotion == "happy":

        return (
            "You seem to have something positive going on. 😊 "
            "What's making you feel good?"
        )

    if emotion == "surprise":

        return (
            "Something seems to have caught you off guard. "
            "What happened?"
        )

    # ========================================================
    # GENERIC FALLBACK
    # ========================================================

    if word_count(text) <= 3:

        return (
            "I'm listening. 💙 Can you tell me a little more "
            "about that?"
        )

    return (
        "I understand. 💙 Tell me a little more about what "
        "happened and how it made you feel."
    )


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
        expression and then gives you a chance to explain
        how you actually feel.
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
        ⚠️ Facial-expression AI is only an estimate.
        It cannot truly know what someone is feeling.
        Your own answer is more important than the camera.
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
                    emotion_model
                )

                if emotion is None:

                    st.error(
                        "I couldn't detect an expression."
                    )

                else:

                    st.session_state.first_emotion = emotion

                    st.session_state.first_confidence = (
                        confidence
                    )

                    emoji = EMOTION_EMOJIS.get(
                        emotion,
                        "🙂"
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
                            use_container_width=True
                        ):

                            st.session_state.actual_emotion = (
                                emotion
                            )

                            st.session_state.stage = (
                                "conversation"
                            )

                            st.experimental_rerun()

                    with col2:

                        if st.button(
                            "❌ No",
                            use_container_width=True
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
        choices
    )

    if st.button(
        "Continue 💙",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.actual_emotion = (
            selected.lower()
        )

        st.session_state.stage = "conversation"

        st.experimental_rerun()


# ============================================================
# CONVERSATION
# ============================================================

elif st.session_state.stage == "conversation":

    emotion = st.session_state.actual_emotion

    emoji = EMOTION_EMOJIS.get(
        emotion,
        "💙"
    )

    st.title("💬 Talk to EmotiCare")

    st.caption(
        f"{emoji} We're checking in about how you're feeling."
    )

    # --------------------------------------------------------
    # INITIAL QUESTION
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
                "content": opening
            }
        )

    # --------------------------------------------------------
    # DISPLAY CHAT
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
    # CHAT FORM
    # --------------------------------------------------------

    with st.form(
        "chat_form",
        clear_on_submit=True
    ):

        user_message = st.text_input(
            "Tell me what's going on..."
        )

        send = st.form_submit_button(
            "Send 💬",
            use_container_width=True
        )

    if send:

        user_message = clean_text(
            user_message
        )

        if user_message:

            # Detect the actual situation BEFORE generating
            # the reply.
            detected_topic = detect_situation(
                user_message
            )

            if detected_topic is not None:

                st.session_state.conversation_topic = (
                    detected_topic
                )

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_message
                }
            )

            st.session_state.conversation_count += 1

            response = generate_response(
                user_message,
                emotion,
                st.session_state.messages
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

            st.experimental_rerun()

        else:

            st.warning(
                "Write something first 🙂"
            )

    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    st.divider()

    st.write(
        "When you're ready, let's check in one more time."
    )

    if st.button(
        "💙 How do you feel now?",
        use_container_width=True,
        type="primary"
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

        EmotiCare will make another facial-expression
        estimate so you can compare the two check-ins.
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
                    emotion_model
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

            That doesn't necessarily mean you feel the same.
            Facial-expression AI can make mistakes.
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
    # HUMAN CHECK
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
            "😔 I still don't feel good"
        ]
    )

    if st.button(
        "Finish 💙",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.final_feeling = (
            final_feeling
        )

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
        use_container_width=True
    ):

        reset_app()
