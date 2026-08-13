import streamlit as st
from PIL import Image
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


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

if "stage" not in st.session_state:
    st.session_state.stage = "welcome"

if "first_emotion" not in st.session_state:
    st.session_state.first_emotion = None

if "first_confidence" not in st.session_state:
    st.session_state.first_confidence = 0.0

if "second_emotion" not in st.session_state:
    st.session_state.second_emotion = None

if "second_confidence" not in st.session_state:
    st.session_state.second_confidence = 0.0

if "actual_emotion" not in st.session_state:
    st.session_state.actual_emotion = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "final_feeling" not in st.session_state:
    st.session_state.final_feeling = None


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
# CHAT MODEL
# ============================================================

@st.cache_resource
def load_chat_model():

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    model.eval()

    return tokenizer, model


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
    "neutral": "😐"
}


# ============================================================
# EMOTION ANALYSIS
# ============================================================

def analyze_emotion(photo, emotion_model):

    image = Image.open(
        photo
    ).convert("RGB")

    results = emotion_model(image)

    if not results:
        return None, 0.0

    best = results[0]

    emotion = best["label"].lower()

    confidence = float(
        best["score"]
    ) * 100

    return emotion, confidence


# ============================================================
# REAL AI RESPONSE
# ============================================================

def generate_response(
    tokenizer,
    model,
    emotion,
    messages
):

    system_prompt = f"""
You are EmotiCare, a friendly and supportive
conversation companion for a school project.

The facial-expression AI estimated that the user's
expression may be "{emotion}".

The facial-expression result is NOT guaranteed to
represent the user's real emotion.

Your job is to have a natural conversation with the user.

IMPORTANT RULES:

- Listen to what the user actually says.
- Respond to their latest message.
- Remember previous messages.
- Ask natural follow-up questions when appropriate.
- Do not use a fixed list of responses.
- Do not repeat the same response.
- Do not force the user to talk about the detected emotion.
- Do not claim that you know exactly how they feel.
- Do not diagnose mental health conditions.
- Do not pretend to be a doctor or therapist.
- Be warm, calm and respectful.
- Keep replies short enough for a chat application.
- Usually respond with 1 to 3 sentences.
- If the user says they are okay, accept that.
- If the user says they are happy, do not insist they are sad.
- The user's own description of their feelings is more
  important than the camera prediction.

The goal is to help the user reflect on how they feel.
"""

    chat_messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    for message in messages:

        chat_messages.append(
            {
                "role": message["role"],
                "content": message["content"]
            }
        )

    prompt = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return response


# ============================================================
# WELCOME SCREEN
# ============================================================

if st.session_state.stage == "welcome":

    st.title("💙 EmotiCare AI")

    st.subheader(
        "A little space to check in with yourself."
    )

    st.write(
        """
        EmotiCare uses AI to estimate facial expressions
        and then lets you have a real AI conversation
        about how you're feeling.
        """
    )

    st.info(
        """
        📸 Camera permission

        Your camera is only used when you choose to
        take a photo for the emotion check.
        """
    )

    st.warning(
        """
        ⚠️ Important

        Facial-expression AI cannot know exactly how
        someone feels.

        It only estimates the expression visible
        in the photo. Your own answer is more important.
        """
    )

    if st.button(
        "💙 Start Check-in",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.stage = "first_scan"

        st.rerun()


# ============================================================
# FIRST CAMERA SCAN
# ============================================================

elif st.session_state.stage == "first_scan":

    st.title("📸 First Emotion Check")

    st.write(
        "Take a photo so the AI can estimate your facial expression."
    )

    photo = st.camera_input(
        "Take your first photo"
    )

    if photo:

        with st.spinner(
            "Loading AI models... This may take a little while the first time."
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

                    st.session_state.first_confidence = confidence

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
                        f"Does **{emotion}** match how you actually feel?"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        if st.button(
                            "✅ Yes",
                            use_container_width=True
                        ):

                            st.session_state.actual_emotion = emotion

                            st.session_state.stage = "conversation"

                            st.rerun()

                    with col2:

                        if st.button(
                            "❌ No",
                            use_container_width=True
                        ):

                            st.session_state.stage = "correct_emotion"

                            st.rerun()

            except Exception as error:

                st.error(
                    "The emotion AI could not analyze the photo."
                )

                st.exception(error)


# ============================================================
# USER CORRECTS AI
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
        "Neutral"
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

        st.session_state.actual_emotion = selected.lower()

        st.session_state.stage = "conversation"

        st.rerun()


# ============================================================
# REAL AI CONVERSATION
# ============================================================

elif st.session_state.stage == "conversation":

    st.title("💬 Talk to EmotiCare")

    # Load the language model only when needed
    with st.spinner(
        "Starting the conversation AI..."
    ):

        try:

            tokenizer, chat_model = load_chat_model()

        except Exception as error:

            st.error(
                "The conversation AI could not load."
            )

            st.exception(error)

            st.stop()

    # --------------------------------------------------------
    # FIRST MESSAGE
    # --------------------------------------------------------

    if len(st.session_state.messages) == 0:

        opening = (
            "I'd like to hear what's on your mind. "
            f"The camera estimated a possible "
            f"{st.session_state.actual_emotion} expression, "
            "but you know yourself better. "
            "What is going on?"
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

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------

    user_message = st.chat_input(
        "Tell me what's going on..."
    )

    if user_message:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        with st.spinner(
            "EmotiCare is thinking..."
        ):

            try:

                response = generate_response(
                    tokenizer,
                    chat_model,
                    st.session_state.actual_emotion,
                    st.session_state.messages
                )

            except Exception as error:

                st.error(
                    "The conversation AI had trouble generating a response."
                )

                st.exception(error)

                response = (
                    "I'm having a little trouble right now. "
                    "Could you tell me a little more?"
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        st.rerun()

    # --------------------------------------------------------
    # FINISH CONVERSATION
    # --------------------------------------------------------

    st.divider()

    st.write(
        "Ready to check in one more time?"
    )

    if st.button(
        "💙 How do I feel now?",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.stage = "second_scan"

        st.rerun()


# ============================================================
# SECOND CAMERA SCAN
# ============================================================

elif st.session_state.stage == "second_scan":

    st.title("💙 How do you feel now?")

    st.write(
        """
        Take one more photo.

        EmotiCare will compare the facial-expression
        estimates from before and now.
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

                    st.session_state.second_confidence = confidence

                    st.session_state.stage = "result"

                    st.rerun()

            except Exception as error:

                st.error(
                    "The second emotion scan failed."
                )

                st.exception(error)


# ============================================================
# RESULT
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

    if first != second:

        st.success(
            f"""
            The AI's estimated facial expression changed
            from **{first}** to **{second}**.
            """
        )

    else:

        st.info(
            f"""
            The AI estimated a similar facial expression
            both times: **{second}**.
            """
        )

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

        st.session_state.final_feeling = final_feeling

        st.session_state.stage = "goodbye"

        st.rerun()


# ============================================================
# GOODBYE
# ============================================================

elif st.session_state.stage == "goodbye":

    st.title("💙 Thank you for checking in")

    feeling = st.session_state.final_feeling

    if "better" in feeling.lower():

        st.success(
            """
            I'm glad you're feeling better. 💙

            Remember to be kind to yourself.
            """

        )

    elif "same" in feeling.lower():

        st.info(
            """
            That's okay. 💙

            You don't have to feel better immediately.
            Give yourself some time.
            """
        )

    else:

        st.info(
            """
            I'm sorry you're still having a difficult moment.

            Consider talking to someone you trust if you
            need some extra support. 💙
            """
        )

    st.write(
        "Thank you for spending a moment with EmotiCare."
    )

    if st.button(
        "🔄 Start Again",
        use_container_width=True
    ):

        keys_to_clear = [
            "stage",
            "first_emotion",
            "first_confidence",
            "second_emotion",
            "second_confidence",
            "actual_emotion",
            "messages",
            "final_feeling"
        ]

        for key in keys_to_clear:

            if key in st.session_state:

                del st.session_state[key]

        st.rerun()
