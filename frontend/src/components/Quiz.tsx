import React, { useState } from 'react';
import { CheckCircle, XCircle } from 'lucide-react';

const Quiz = () => {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [score, setScore] = useState(0);
  const [showResult, setShowResult] = useState(false);

  const questions = [
    {
      question: "What is the sign for 'Hello'?",
      options: [
        "Wave hand side to side",
        "Touch forehead",
        "Cross arms",
        "Point up"
      ],
      correct: 0
    },
    {
      question: "Which hand position represents 'Thank you'?",
      options: [
        "Thumbs up",
        "Touch lips and move forward",
        "Wave hand",
        "Clap hands"
      ],
      correct: 1
    },
    {
      question: "What is the correct sign for 'Please'?",
      options: [
        "Rub chest clockwise",
        "Point to mouth",
        "Wave hand",
        "Touch heart"
      ],
      correct: 0
    }
  ];

  const handleAnswer = (selectedOption: number) => {
    if (selectedOption === questions[currentQuestion].correct) {
      setScore(score + 1);
    }

    if (currentQuestion + 1 < questions.length) {
      setCurrentQuestion(currentQuestion + 1);
    } else {
      setShowResult(true);
    }
  };

  const restartQuiz = () => {
    setCurrentQuestion(0);
    setScore(0);
    setShowResult(false);
  };

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-white">Test Your Knowledge</h1>
        <p className="mt-2 text-gray-600">Challenge yourself with our sign language quiz</p>
      </div>

      {!showResult ? (
        <div className="card space-y-6">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500">
              Question {currentQuestion + 1} of {questions.length}
            </span>
            <span className="text-sm text-gray-500">
              Score: {score}
            </span>
          </div>

          <h2 className="text-xl font-semibold">
            {questions[currentQuestion].question}
          </h2>

          <div className="space-y-3">
            {questions[currentQuestion].options.map((option, index) => (
              <button
                key={index}
                className="w-full text-left p-4 rounded-lg border hover:bg-blue-50 
                         hover:border-blue-200 transition-colors duration-200"
                onClick={() => handleAnswer(index)}
              >
                {option}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="card text-center space-y-6">
          <h2 className="text-2xl font-bold">Quiz Complete!</h2>
          <div className="text-6xl font-bold text-blue-600">
            {Math.round((score / questions.length) * 100)}%
          </div>
          <p className="text-gray-600">
            You got {score} out of {questions.length} questions correct
          </p>
          {score === questions.length ? (
            <div className="flex items-center justify-center text-green-500">
              <CheckCircle className="h-6 w-6 mr-2" />
              Perfect Score!
            </div>
          ) : (
            <div className="flex items-center justify-center text-yellow-500">
              <XCircle className="h-6 w-6 mr-2" />
              Keep practicing!
            </div>
          )}
          <button
            className="btn-primary"
            onClick={restartQuiz}
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
};

export default Quiz;