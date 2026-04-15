async function predict() {
  const fileInput = document.getElementById("fileInput");
  const resultDiv = document.getElementById("result");

  if (!fileInput.files.length) {
    resultDiv.innerText = "Please upload a CSV file";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const response = await fetch("http://127.0.0.1:8000/predict-file", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    resultDiv.innerText = "Prediction: " + JSON.stringify(data.prediction);
  } catch (err) {
    resultDiv.innerText = "Error connecting to API";
  }
}