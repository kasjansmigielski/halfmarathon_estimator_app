### Application description:
The aim of the project was to create an application that would use a regression algorithm to train models and would be able to predict (based on previously trained data) the time in which a user would run a halfmarathon - by providing specific data.

### Main functionalities:
* allowing the user to enter data freely (without any appropriate conversion of the record) -> the LLM model used extracts data from the user into a JSON structure and prepares it for use by the regression model,
* simple functionality allows for the final estimation of the time to run a half marathon - using the trained best regression model,
* the LLM model is connected to Langfuse to track the model's life cycle.

### Dependencies:
* streamlit,
* pycaret,
* langfuse,
* openai,
* pandas,
* instructor,
* pydantic
* python-dotenv.

### Maintenance:
* aplikacja wykorzystuje GitHub Actions do automatycznego utrzymania aktywności (zapobieganie hibernacji Streamlit po 12h nieaktywności) poprzez regularne wykonywanie pustych commitów.

### Result:
The application is publicly deployed at the link: https://halfmarathon-estimator.streamlit.app/
