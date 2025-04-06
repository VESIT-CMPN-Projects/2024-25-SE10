import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, NavLink } from 'react-router-dom';
import { Camera, Type, BookOpen, GraduationCap, Users, BookMarked, Menu, X } from 'lucide-react';
import Home from './components/Home';
import Learn from './components/Learn';
import ISL from './components/ISL';
import Quiz from './components/Quiz';
import Blog from './components/Blog';
import Community from './components/Community';
import Resources from './components/Resources';
import ThemeProvider, { ThemeToggle } from './components/ThemeProvider';

function App() {
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);

  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);

  return (
    <ThemeProvider>
      <Router>
        <div className="min-h-screen">
          {/* Decorative Elements */}
          <div className="fixed top-40 left-10 w-64 h-64 bg-indigo-300 dark:bg-indigo-800 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
          <div className="fixed top-20 right-20 w-80 h-80 bg-purple-300 dark:bg-purple-800 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
          <div className="fixed bottom-40 right-10 w-72 h-72 bg-pink-300 dark:bg-pink-800 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>
          
          <nav className="bg-white dark:bg-dark-card bg-opacity-80 dark:bg-opacity-80 backdrop-blur-md sticky top-0 z-50 shadow-sm border-b border-gray-100 dark:border-gray-800">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between h-16">
                <div className="flex items-center">
                  <Link to="/" className="flex items-center space-x-2 group">
                    <div className="p-2 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg text-white 
                        transition-all duration-300 group-hover:shadow-md group-hover:shadow-indigo-300 dark:group-hover:shadow-indigo-800">
                      <Type className="h-6 w-6" />
                    </div>
                    <span className="text-xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 dark:from-indigo-400 dark:to-purple-400 bg-clip-text text-transparent">
                      SignBridge
                    </span>
                  </Link>
                </div>

                {/* Desktop Navigation */}
                <div className="hidden md:flex items-center space-x-4">
                  <NavLink to="/learn" className="nav-link">
                    <BookOpen className="h-5 w-5 inline-block mr-1" />
                    Learn
                  </NavLink>
                  <NavLink to="/isl" className="nav-link">
                    <Camera className="h-5 w-5 inline-block mr-1" />
                    ISL
                  </NavLink>
                  <NavLink to="/quiz" className="nav-link">
                    <GraduationCap className="h-5 w-5 inline-block mr-1" />
                    Quiz
                  </NavLink>
                  <NavLink to="/blog" className="nav-link">
                    <BookMarked className="h-5 w-5 inline-block mr-1" />
                    Blog
                  </NavLink>
                  <NavLink to="/community" className="nav-link">
                    <Users className="h-5 w-5 inline-block mr-1" />
                    Community
                  </NavLink>
                  <NavLink to="/resources" className="nav-link">
                    <BookMarked className="h-5 w-5 inline-block mr-1" />
                    Resources
                  </NavLink>
                  
                  <div className="pl-2 border-l border-gray-200 dark:border-gray-700">
                    <ThemeToggle />
                  </div>
                </div>

                {/* Mobile menu button */}
                <div className="md:hidden flex items-center space-x-2">
                  <ThemeToggle />
                  <button
                    onClick={toggleMenu}
                    className="inline-flex items-center justify-center p-2 rounded-md text-indigo-500 dark:text-indigo-400 hover:text-indigo-600 dark:hover:text-indigo-300 
                      hover:bg-indigo-50 dark:hover:bg-indigo-900/30 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500 dark:focus:ring-indigo-400
                      transition-all duration-200"
                  >
                    {isMenuOpen ? (
                      <X className="block h-6 w-6" />
                    ) : (
                      <Menu className="block h-6 w-6" />
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Mobile Navigation */}
            {isMenuOpen && (
              <div className="md:hidden bg-white dark:bg-dark-card bg-opacity-95 dark:bg-opacity-95 backdrop-blur-sm border-t border-gray-100 dark:border-gray-800 shadow-md rounded-b-xl">
                <div className="px-2 pt-2 pb-3 space-y-1">
                  <NavLink to="/learn" className="nav-link block" onClick={toggleMenu}>
                    <BookOpen className="h-5 w-5 inline-block mr-1" />
                    Learn
                  </NavLink>
                  <NavLink to="/isl" className="nav-link block" onClick={toggleMenu}>
                    <Camera className="h-5 w-5 inline-block mr-1" />
                    ISL
                  </NavLink>
                  <NavLink to="/quiz" className="nav-link block" onClick={toggleMenu}>
                    <GraduationCap className="h-5 w-5 inline-block mr-1" />
                    Quiz
                  </NavLink>
                  <NavLink to="/blog" className="nav-link block" onClick={toggleMenu}>
                    <BookMarked className="h-5 w-5 inline-block mr-1" />
                    Blog
                  </NavLink>
                  <NavLink to="/community" className="nav-link block" onClick={toggleMenu}>
                    <Users className="h-5 w-5 inline-block mr-1" />
                    Community
                  </NavLink>
                  <NavLink to="/resources" className="nav-link block" onClick={toggleMenu}>
                    <BookMarked className="h-5 w-5 inline-block mr-1" />
                    Resources
                  </NavLink>
                </div>
              </div>
            )}
          </nav>

          <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/learn" element={<Learn />} />
              <Route path="/isl" element={<ISL />} />
              <Route path="/quiz" element={<Quiz />} />
              <Route path="/blog" element={<Blog />} />
              <Route path="/community" element={<Community />} />
              <Route path="/resources" element={<Resources />} />
            </Routes>
          </main>

          <footer className="bg-gradient-to-r from-indigo-600 to-purple-600 dark:from-indigo-800 dark:to-purple-800 text-white mt-16 relative z-10">
            <div className="absolute inset-0 overflow-hidden">
              <div className="absolute left-0 w-full h-10 bg-white transform -translate-y-1/2 rounded-full opacity-10 blur-md"></div>
            </div>
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative">
              <div className="grid md:grid-cols-3 gap-8">
                <div>
                  <h3 className="text-xl font-bold text-white mb-4">SignBridge</h3>
                  <p className="text-indigo-100">
                    Bridging communication gaps between the deaf and hearing communities through technology.
                  </p>
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white mb-4">Quick Links</h3>
                  <ul className="space-y-2">
                    <li><Link to="/learn" className="text-indigo-100 hover:text-white transition-colors duration-200">Learn</Link></li>
                    <li><Link to="/isl" className="text-indigo-100 hover:text-white transition-colors duration-200">ISL</Link></li>
                    <li><Link to="/resources" className="text-indigo-100 hover:text-white transition-colors duration-200">Resources</Link></li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white mb-4">Connect</h3>
                  <p className="text-indigo-100 mb-2">
                    Join our community of sign language enthusiasts
                  </p>
                  <Link to="/community" className="bg-white text-indigo-600 px-4 py-2 rounded-lg font-medium
                    hover:bg-opacity-90 transition-colors duration-200 inline-block shadow-md hover:shadow-lg">
                    Join Community
                  </Link>
                </div>
              </div>
              <div className="border-t border-indigo-400 mt-8 pt-8 text-center text-indigo-100">
                <p>© {new Date().getFullYear()} SignBridge. All rights reserved.</p>
              </div>
            </div>
          </footer>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;