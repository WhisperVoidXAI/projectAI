const video = document.getElementById("webcam");
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri("models"),
  faceapi.nets.ageGenderNet.loadFromUri("models")
]).then(startVideo);

function startVideo() {
  navigator.mediaDevices.getUserMedia({ video: {} })
    .then(stream => video.srcObject = stream)
    .catch(err => console.error(err));
}
video.addEventListener("playing", () => {
  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withAgeAndGender();

    if (detections && detections.length > 0) {
      const detection = detections[0];
      document.getElementById(detection.gender).checked = true;
      document.getElementById("age").value = Math.round(detection.age);
    }
  }, 1000);
});
