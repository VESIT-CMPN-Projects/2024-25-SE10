import React, { useState, useRef, useCallback, useEffect } from "react";
import Webcam from "react-webcam";
import { Camera, Type, RefreshCw } from "lucide-react";

// Both endpoints are now served from port 5005
const DETECTION_ENDPOINT = "http://127.0.0.1:5005/predict";
const TRANSLATION_ENDPOINT = "http://127.0.0.1:5000/process_sentence";

// Video constraints for high resolution preview
const videoConstraints = {
  width: { ideal: 1280 },
  height: { ideal: 720 },
  facingMode: "user",
};

// Detection capture options for lower resolution
const detectionCaptureOptions = {
  width: 640,
  height: 360,
  quality: 0.7,
};

const ISL = () => {
  const [mode, setMode] = useState<"sign-to-text" | "text-to-sign">(
    "sign-to-text"
  );
  const [text, setText] = useState("");
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionResult, setDetectionResult] = useState("No sign detected...");
  const [preview, setPreview] = useState(""); // To hold the MediaPipe-processed image preview
  const [videoUrl, setVideoUrl] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const webcamRef = useRef<Webcam>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const handleStartDetection = () => {
    setIsDetecting(true);
  };

  const handleStopDetection = () => {
    setIsDetecting(false);
    setDetectionResult("No sign detected...");
    setPreview("");
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  // Text-to-sign translation handler
  const handleTranslate = async () => {
    try {
      const response = await fetch(TRANSLATION_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentence: text }),
      });
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      const buffer = await response.arrayBuffer();
      const blob = new Blob([buffer], { type: "video/mp4" });
      const url = URL.createObjectURL(blob);

      if (videoRef.current) {
        videoRef.current.src = url;
        videoRef.current.load();
      }
      setVideoUrl(url);
    } catch (error) {
      console.error("Error translating sentence:", error);
    }
  };

  // Capture image from webcam and send it to the detection endpoint using lower-res options
  const captureAndDetect = useCallback(async () => {
    if (webcamRef.current && isDetecting) {
      const imageSrc = webcamRef.current.getScreenshot(detectionCaptureOptions);
      if (imageSrc) {
        try {
          const response = await fetch(DETECTION_ENDPOINT, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imageSrc }),
          });
          console.log("Detection response:", response);
          if (response.ok) {
            const data = await response.json();
            setDetectionResult(`${data.label} (${data.confidence})`);
            if (data.preview) {
              setPreview(data.preview);
            }
          } else {
            console.error(
              "Error sending frame for detection:",
              response.status
            );
            setDetectionResult("Error during detection");
          }
        } catch (error) {
          console.error("Error sending frame for detection:", error);
          setDetectionResult("Error during detection");
        }
      }
    }
  }, [isDetecting]);

  useEffect(() => {
    if (isDetecting) {
      // Increase interval to 250ms for less frequent requests
      intervalRef.current = setInterval(captureAndDetect, 250);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isDetecting, captureAndDetect]);

  return (
    <div className="space-y-8">
      <div className="flex justify-center space-x-4">
        <button
          className={`nav-link ${mode === "sign-to-text" ? "active" : ""}`}
          onClick={() => setMode("sign-to-text")}
        >
          <Camera className="h-5 w-5 inline-block mr-1" />
          Sign to Text
        </button>
        <button
          className={`nav-link ${mode === "text-to-sign" ? "active" : ""}`}
          onClick={() => setMode("text-to-sign")}
        >
          <Type className="h-5 w-5 inline-block mr-1" />
          Text to Sign
        </button>
      </div>

      <div className="card">
        {mode === "sign-to-text" ? (
          <div className="space-y-4">
            <div className="relative h-150 overflow-hidden rounded-lg">
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                style={{ objectFit: "cover" }}
              />
              <button
                className={`absolute bottom-4 right-4 btn-primary ${
                  isDetecting ? "bg-red-600 hover:bg-red-700" : ""
                }`}
                onClick={
                  isDetecting ? handleStopDetection : handleStartDetection
                }
              >
                {isDetecting ? "Stop Detection" : "Start Detection"}
              </button>
              {/* Display MediaPipe-processed preview in the corner */}
              {preview && (
                <img
                  src={preview}
                  alt="MediaPipe Preview"
                  style={{
                    position: "absolute",
                    bottom: "10px",
                    left: "10px",
                    width: "200px",
                    border: "2px solid #fff",
                    borderRadius: "4px",
                  }}
                />
              )}
            </div>

            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-2">Detection Result</h3>
              <p className="text-gray-600">{detectionResult}</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div>
              <textarea
                className="w-full p-4 border rounded-lg text-2xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-dark-card dark:text-white"
                rows={1}
                placeholder="Enter text to translate to sign language..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
            </div>
            <button className="btn-primary" onClick={handleTranslate}>
              <RefreshCw className="h-5 w-5 inline-block mr-1" />
              Translate
            </button>
            <div className="bg-gray-50 p-4 rounded-lg h-90 flex items-center justify-center dark:bg-dark-card dark:text-white overflow-hidden">
              {videoUrl ? (
                <video ref={videoRef} width="100%" height="100%" controls>
                  <source src={videoUrl} type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
              ) : (
                <img
                  src="src/placeholder.jfif"
                  alt="Placeholder"
                  className="w-full h-full object-cover rounded-lg"
                />
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ISL;
