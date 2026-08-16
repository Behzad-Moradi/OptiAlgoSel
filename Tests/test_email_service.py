import pytest
from API.services.email_service import send_email


def test_send_email_success(mocker):

    mock_smtp = mocker.patch("API.services.email_service.smtplib.SMTP_SSL")
    smtp_instance = mock_smtp.return_value.__enter__.return_value
    user_email = "test@example.com"
    predicted_algorithms = ["CMAES", "XGBoost"]
    result = send_email(user_email, predicted_algorithms)
    assert result == "Email sent successfully."
    mock_smtp.assert_called_once_with("smtp.gmail.com", 465)
    smtp_instance.login.assert_called_once()
    smtp_instance.send_message.assert_called_once()


def test_send_email_to_correct_recipient(mocker):

    mock_smtp = mocker.patch("API.services.email_service.smtplib.SMTP_SSL")
    smtp_instance = mock_smtp.return_value.__enter__.return_value
    user_email = "test@example.com"
    predicted_algorithms = ["CMAES", "XGBoost"]
    send_email(user_email, predicted_algorithms)
    message = smtp_instance.send_message.call_args.args[0]
    assert message["To"] == user_email
    assert message["Subject"] == "OptiAlgoSel Prediction Results"
    assert "CMAES" in message.get_content()
    assert "XGBoost" in message.get_content()


def test_send_email_failure(mocker):

    mock_smtp = mocker.patch("API.services.email_service.smtplib.SMTP_SSL")
    smtp_instance = mock_smtp.return_value.__enter__.return_value
    smtp_instance.login.side_effect = Exception("SMTP connection failed")
    user_email = "test@example.com"
    predicted_algorithms = ["CMAES"]
    with pytest.raises(RuntimeError, match="Failed to send email"):
        send_email(user_email, predicted_algorithms)