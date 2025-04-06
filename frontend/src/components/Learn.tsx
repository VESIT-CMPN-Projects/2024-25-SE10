import React from 'react';
import { BookOpen, Star, Clock, ArrowRight, Trophy, PlayCircle } from 'lucide-react';

const lessons = [
  {
    id: 1,
    title: 'Basic Greetings',
    description: 'Learn common greetings and introductions in sign language.',
    difficulty: 'Beginner',
    duration: '15 mins',
    image: 'https://images.unsplash.com/photo-1516533075015-a3838414c3ca?auto=format&fit=crop&q=80&w=500',
  },
  {
    id: 2,
    title: 'Numbers and Counting',
    description: 'Master numbers from 1-100 in sign language.',
    difficulty: 'Beginner',
    duration: '20 mins',
    image: 'https://images.unsplash.com/photo-1509228468518-180dd4864904?auto=format&fit=crop&q=80&w=500',
  },
  {
    id: 3,
    title: 'Common Phrases',
    description: 'Learn everyday phrases and expressions.',
    difficulty: 'Intermediate',
    duration: '25 mins',
    image: 'https://images.unsplash.com/photo-1521791136064-7986c2920216?auto=format&fit=crop&q=80&w=500',
  },
];

const Learn = () => {
  return (
    <div className="space-y-8">
      <div className="hero-gradient text-center p-10 rounded-3xl relative overflow-hidden">
        {/* Decorative elements */}
        <div className="absolute top-0 left-0 w-full h-full bg-dot-pattern bg-dot-md opacity-30 z-0"></div>
        <div className="absolute -top-10 -right-10 w-20 h-20 bg-purple-300 dark:bg-purple-800 rounded-full mix-blend-multiply filter blur-xl opacity-30"></div>
        <div className="absolute -bottom-10 -left-10 w-20 h-20 bg-indigo-300 dark:bg-indigo-800 rounded-full mix-blend-multiply filter blur-xl opacity-30"></div>
        
        <div className="relative z-10">
          <h1 className="text-3xl md:text-4xl font-bold animate-gradient-x bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-500 dark:from-indigo-400 dark:via-purple-400 dark:to-pink-400 bg-clip-text text-transparent">
            Learn Sign Language
          </h1>
          <p className="mt-4 text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Start your journey to mastering sign language with our interactive lessons designed for all skill levels.
          </p>
        </div>
      </div>

      <h2 className="section-title">Featured Lessons</h2>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mt-8">
        {lessons.map((lesson, index) => (
          <div key={lesson.id} className="interactive-card group overflow-hidden gradient-border">
            <div className="relative">
              <img
                src={lesson.image}
                alt={lesson.title}
                className="w-full h-48 object-cover rounded-t-xl -mx-6 -mt-6 mb-4 group-hover:scale-105 transition-transform duration-500"
              />
              <div className="absolute top-2 right-2 py-1 px-3 bg-gradient-to-r from-indigo-600 to-purple-600 dark:from-indigo-500 dark:to-purple-500 rounded-full text-white text-xs font-medium">
                {lesson.difficulty}
              </div>
              <div className="absolute bottom-4 right-4 p-2 rounded-full bg-white/80 dark:bg-dark-card/80 backdrop-blur-sm group-hover:bg-indigo-600 dark:group-hover:bg-indigo-600 text-indigo-600 dark:text-indigo-400 group-hover:text-white dark:group-hover:text-white transition-all duration-300">
                <PlayCircle className="h-6 w-6" />
              </div>
            </div>
            <h3 className="text-xl font-semibold mb-2 gradient-text">{lesson.title}</h3>
            <p className="text-gray-600 dark:text-gray-300 mb-4">{lesson.description}</p>
            <div className="flex items-center text-sm text-gray-500 dark:text-gray-400 mb-4">
              <div className="flex items-center">
                <Clock className="h-4 w-4 mr-1 text-indigo-500 dark:text-indigo-400" />
                {lesson.duration}
              </div>
            </div>
            <button className="btn-primary w-full mt-2 group-hover:shadow-indigo-200/50 dark:group-hover:shadow-indigo-800/30 flex items-center justify-center animate-float animation-delay-${index * 2000}">
              Start Lesson
              <ArrowRight className="h-4 w-4 ml-2 group-hover:translate-x-1 transition-transform duration-300" />
            </button>
          </div>
        ))}
      </div>

      <div className="glass rounded-xl p-8 mt-12 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-300 dark:bg-indigo-800 rounded-full mix-blend-multiply filter blur-xl opacity-20"></div>
        <div className="absolute bottom-0 left-0 w-32 h-32 bg-purple-300 dark:bg-purple-800 rounded-full mix-blend-multiply filter blur-xl opacity-20"></div>
        
        <h2 className="text-2xl font-semibold mb-6 gradient-text">Your Progress</h2>
        <div className="bg-white dark:bg-dark-card bg-opacity-90 dark:bg-opacity-80 backdrop-blur-sm rounded-lg p-6 shadow-md relative z-10">
          <div className="flex items-center justify-between mb-3">
            <span className="text-gray-700 dark:text-gray-300 font-medium">Overall Progress</span>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600 dark:from-indigo-400 dark:to-purple-400 font-bold text-lg">35%</span>
          </div>
          <div className="w-full bg-gray-100 dark:bg-gray-800 rounded-full h-3">
            <div className="bg-gradient-to-r from-indigo-600 to-purple-600 dark:from-indigo-500 dark:to-purple-500 h-3 rounded-full shadow-sm animate-pulse-soft" style={{ width: '35%' }}></div>
          </div>
          
          <div className="mt-6 grid grid-cols-2 gap-4">
            <div className="glass-card animate-float">
              <div className="flex items-center space-x-3">
                <Trophy className="h-6 w-6 text-indigo-500 dark:text-indigo-400" />
                <div>
                  <h3 className="text-lg font-medium mb-1 text-gray-800 dark:text-gray-200">Lessons Completed</h3>
                  <p className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600 dark:from-indigo-400 dark:to-purple-400">7</p>
                </div>
              </div>
            </div>
            <div className="glass-card animate-float animation-delay-2000">
              <div className="flex items-center space-x-3">
                <Star className="h-6 w-6 text-amber-500" />
                <div>
                  <h3 className="text-lg font-medium mb-1 text-gray-800 dark:text-gray-200">Current Streak</h3>
                  <p className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600 dark:from-indigo-400 dark:to-purple-400">5 days</p>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-indigo-50 dark:bg-indigo-900/30 rounded-lg border border-indigo-100 dark:border-indigo-800">
            <h3 className="text-lg font-medium mb-2 text-indigo-600 dark:text-indigo-400">Recommended Next Steps</h3>
            <ul className="space-y-2">
              <li className="flex items-center text-gray-700 dark:text-gray-300">
                <ArrowRight className="h-4 w-4 mr-2 text-indigo-500 dark:text-indigo-400" />
                Continue "Common Phrases" lesson
              </li>
              <li className="flex items-center text-gray-700 dark:text-gray-300">
                <ArrowRight className="h-4 w-4 mr-2 text-indigo-500 dark:text-indigo-400" />
                Practice "Numbers and Counting" with quiz
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Learn;