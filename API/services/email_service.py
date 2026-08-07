import smtplib
from email.message import EmailMessage
from API.config import SENDER_EMAIL, APP_PASSWORD


def send_email(user_email: str, predicted_algorithms: list):


    message = EmailMessage()

    message["Subject"] = "OptiAlgoSel Prediction Results"
    message["From"] = SENDER_EMAIL
    message["To"] = user_email

    algorithms = "\n".join(
        f"{i+1}. {alg}"
        for i, alg in enumerate(predicted_algorithms)
    )

    message.set_content(
        f"""Hello,

                Your prediction request has been completed successfully.

                Recommended optimization algorithm(s):

                {algorithms}

                Thank you for using OptiAlgoSel.

                Regards,
                OptiAlgoSel Team
                """
                    )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(message)

        return "Email sent successfully."

    except Exception as e:
        raise RuntimeError(f"Failed to send email: {e}")