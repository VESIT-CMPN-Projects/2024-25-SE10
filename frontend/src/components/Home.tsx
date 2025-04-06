import React from 'react';
import { Link } from 'react-router-dom';
import { Camera, Type, ArrowRight, Users, BookOpen, Award, Sparkles, Heart, Globe } from 'lucide-react';

const Home = () => {
  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <section className="hero-gradient text-center space-y-8 my-8 relative overflow-hidden">
        {/* Decorative elements */}
        <div className="absolute top-0 left-0 w-full h-full bg-dot-pattern bg-dot-md opacity-30 z-0"></div>
        <div className="absolute -top-10 -right-10 w-40 h-40 bg-purple-300 dark:bg-purple-800 rounded-full mix-blend-multiply filter blur-xl opacity-30"></div>
        <div className="absolute -bottom-10 -left-10 w-40 h-40 bg-indigo-300 dark:bg-indigo-800 rounded-full mix-blend-multiply filter blur-xl opacity-30"></div>
        
        <div className="relative z-10">
          <h1 className="text-5xl md:text-6xl font-bold animate-gradient-x bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-500 dark:from-indigo-400 dark:via-purple-400 dark:to-pink-400 bg-clip-text text-transparent">
            Bridge the Communication Gap with SignBridge
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto mt-6">
            Transform the way you communicate with our advanced sign language translation platform.
            Learn, practice, and connect with the deaf community.
          </p>
          <div className="flex flex-col sm:flex-row justify-center items-center space-y-4 sm:space-y-0 sm:space-x-4 mt-8">
            <Link to="/learn" className="btn-primary group">
              Start Learning
              <ArrowRight className="inline-block ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform duration-300" />
            </Link>
            <Link to="/community" className="btn-secondary group">
              Join Community
              <Users className="inline-block ml-2 h-5 w-5" />
            </Link>
          </div>
        </div>
        
        {/* Floating elements */}
        <div className="absolute top-1/4 left-1/4 text-indigo-400 dark:text-indigo-300 opacity-40 animate-float">
          <Sparkles className="h-6 w-6" />
        </div>
        <div className="absolute bottom-1/4 right-1/4 text-purple-400 dark:text-purple-300 opacity-40 animate-pulse-soft animation-delay-2000">
          <Sparkles className="h-6 w-6" />
        </div>
        <div className="absolute top-3/4 right-1/3 text-pink-400 dark:text-pink-300 opacity-30 animate-float animation-delay-4000">
          <Heart className="h-5 w-5" />
        </div>
        <div className="absolute bottom-1/3 left-1/3 text-indigo-400 dark:text-indigo-300 opacity-30 animate-pulse-soft animation-delay-6000">
          <Globe className="h-5 w-5" />
        </div>
      </section>

      {/* Features Section */}
      <section>
        <h2 className="section-title mx-auto text-center">Our Features</h2>
        <div className="grid md:grid-cols-2 gap-8 mt-12">
          <div className="interactive-card group">
            <div className="absolute top-0 right-0 w-20 h-20 bg-indigo-100 dark:bg-indigo-900 rounded-full mix-blend-multiply filter blur-xl opacity-30 group-hover:opacity-60 transition-opacity duration-300"></div>
            <div className="flex items-center space-x-4 mb-4 relative z-10">
              <div className="p-3 bg-gradient-to-br from-indigo-100 to-indigo-200 dark:from-indigo-900 dark:to-indigo-800 rounded-lg group-hover:bg-gradient-to-br group-hover:from-indigo-500 group-hover:to-purple-500 transition-all duration-300">
                <Camera className="h-8 w-8 text-indigo-600 dark:text-indigo-400 group-hover:text-white transition-colors duration-300" />
              </div>
              <h3 className="text-xl font-semibold">Sign-to-Text Translation</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-300 relative z-10">
              Use your camera for real-time sign language detection and translation.
              Our advanced AI technology makes communication seamless.
            </p>
          </div>
          <div className="interactive-card group">
            <div className="absolute top-0 right-0 w-20 h-20 bg-purple-100 dark:bg-purple-900 rounded-full mix-blend-multiply filter blur-xl opacity-30 group-hover:opacity-60 transition-opacity duration-300"></div>
            <div className="flex items-center space-x-4 mb-4 relative z-10">
              <div className="p-3 bg-gradient-to-br from-indigo-100 to-indigo-200 dark:from-indigo-900 dark:to-indigo-800 rounded-lg group-hover:bg-gradient-to-br group-hover:from-indigo-500 group-hover:to-purple-500 transition-all duration-300">
                <Type className="h-8 w-8 text-indigo-600 dark:text-indigo-400 group-hover:text-white transition-colors duration-300" />
              </div>
              <h3 className="text-xl font-semibold">Text-to-Sign Translation</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-300 relative z-10">
              Convert text into sign language animations instantly.
              Perfect for learning and practicing sign language.
            </p>
          </div>
          <div className="interactive-card group">
            <div className="absolute top-0 right-0 w-20 h-20 bg-indigo-100 dark:bg-indigo-900 rounded-full mix-blend-multiply filter blur-xl opacity-30 group-hover:opacity-60 transition-opacity duration-300"></div>
            <div className="flex items-center space-x-4 mb-4 relative z-10">
              <div className="p-3 bg-gradient-to-br from-indigo-100 to-indigo-200 dark:from-indigo-900 dark:to-indigo-800 rounded-lg group-hover:bg-gradient-to-br group-hover:from-indigo-500 group-hover:to-purple-500 transition-all duration-300">
                <BookOpen className="h-8 w-8 text-indigo-600 dark:text-indigo-400 group-hover:text-white transition-colors duration-300" />
              </div>
              <h3 className="text-xl font-semibold">Interactive Learning</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-300 relative z-10">
              Learn sign language through interactive lessons, quizzes, and real-time feedback.
              Track your progress and master new signs at your own pace.
            </p>
          </div>
          <div className="interactive-card group">
            <div className="absolute top-0 right-0 w-20 h-20 bg-purple-100 dark:bg-purple-900 rounded-full mix-blend-multiply filter blur-xl opacity-30 group-hover:opacity-60 transition-opacity duration-300"></div>
            <div className="flex items-center space-x-4 mb-4 relative z-10">
              <div className="p-3 bg-gradient-to-br from-indigo-100 to-indigo-200 dark:from-indigo-900 dark:to-indigo-800 rounded-lg group-hover:bg-gradient-to-br group-hover:from-indigo-500 group-hover:to-purple-500 transition-all duration-300">
                <Users className="h-8 w-8 text-indigo-600 dark:text-indigo-400 group-hover:text-white transition-colors duration-300" />
              </div>
              <h3 className="text-xl font-semibold">Community Connection</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-300 relative z-10">
              Join a thriving community of sign language learners and native signers.
              Share experiences, practice together, and make meaningful connections.
            </p>
          </div>
        </div>
      </section>

      {/* Statistics Section */}
      <section className="rounded-2xl p-8 overflow-hidden relative">
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 to-purple-600 opacity-10 animate-gradient-x"></div>
        
        {/* Glass effect card container */}
        <div className="relative z-10 p-2 rounded-xl glass">
          <h2 className="section-title mx-auto text-center mb-12">Our Impact</h2>
          <div className="grid md:grid-cols-3 gap-8 text-center relative z-10">
            <div className="gradient-border animate-float animation-delay-2000">
              <div className="stat-card">
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-600 to-purple-600 dark:from-indigo-700 dark:to-purple-700 opacity-80 animate-gradient-y rounded-xl -z-10"></div>
                <div className="relative z-10">
                  <h4 className="text-4xl font-bold text-white">10K+</h4>
                  <p className="text-indigo-100 mt-2">Active Learners</p>
                  <Award className="h-8 w-8 mx-auto mt-4 text-indigo-200" />
                </div>
              </div>
            </div>
            <div className="gradient-border animate-float">
              <div className="stat-card">
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-600 to-purple-600 dark:from-indigo-700 dark:to-purple-700 opacity-80 animate-gradient-y rounded-xl -z-10"></div>
                <div className="relative z-10">
                  <h4 className="text-4xl font-bold text-white">1000+</h4>
                  <p className="text-indigo-100 mt-2">Sign Language Gestures</p>
                  <Award className="h-8 w-8 mx-auto mt-4 text-indigo-200" />
                </div>
              </div>
            </div>
            <div className="gradient-border animate-float animation-delay-4000">
              <div className="stat-card">
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-600 to-purple-600 dark:from-indigo-700 dark:to-purple-700 opacity-80 animate-gradient-y rounded-xl -z-10"></div>
                <div className="relative z-10">
                  <h4 className="text-4xl font-bold text-white">95%</h4>
                  <p className="text-indigo-100 mt-2">Translation Accuracy</p>
                  <Award className="h-8 w-8 mx-auto mt-4 text-indigo-200" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="glass-card bg-gradient-to-r from-indigo-600 to-purple-600 dark:from-indigo-700 dark:to-purple-700 rounded-2xl p-10 text-center text-white relative overflow-hidden">
        {/* Background decoration */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-white rounded-full mix-blend-overlay opacity-10 transform translate-x-1/2 -translate-y-1/2"></div>
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-white rounded-full mix-blend-overlay opacity-10 transform -translate-x-1/2 translate-y-1/2"></div>
        
        <div className="relative z-10">
          <h2 className="text-3xl font-bold mb-4 text-white">Ready to start your sign language journey?</h2>
          <p className="max-w-2xl mx-auto mb-8 text-indigo-100">
            Join thousands of learners who are bridging communication gaps with SignBridge
          </p>
          <div className="flex flex-col sm:flex-row justify-center items-center space-y-4 sm:space-y-0 sm:space-x-4">
            <Link to="/learn" className="bg-white text-indigo-600 px-8 py-3 rounded-lg font-semibold 
              hover:bg-opacity-90 transition-all duration-300 inline-block shadow-lg hover:shadow-indigo-500/30 hover:-translate-y-1">
              Get Started Now
              <ArrowRight className="inline-block ml-2 h-5 w-5" />
            </Link>
            <Link to="/resources" className="bg-transparent border border-white text-white px-8 py-3 rounded-lg font-semibold 
              hover:bg-white/10 transition-all duration-300 inline-block hover:-translate-y-1">
              Explore Resources
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;