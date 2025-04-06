import React from 'react';
import { Calendar, User, ArrowRight } from 'lucide-react';

const Blog = () => {
  const posts = [
    {
      id: 1,
      title: 'The Evolution of Sign Language in the Digital Age',
      excerpt: 'Explore how technology is transforming the way we learn and use sign language...',
      author: 'Sarah Johnson',
      date: 'March 15, 2024',
      image: 'https://images.unsplash.com/photo-1516321497487-e288fb19713f?auto=format&fit=crop&q=80&w=500',
    },
    {
      id: 2,
      title: 'Breaking Communication Barriers Through Technology',
      excerpt: 'Discover how AI and machine learning are making sign language more accessible...',
      author: 'Michael Chen',
      date: 'March 12, 2024',
      image: 'https://images.unsplash.com/photo-1531498860502-7c67cf02f657?auto=format&fit=crop&q=80&w=500',
    },
    {
      id: 3,
      title: 'The Importance of Sign Language in Modern Society',
      excerpt: 'Understanding why learning sign language is more relevant than ever...',
      author: 'Emma Davis',
      date: 'March 10, 2024',
      image: 'https://images.unsplash.com/photo-1521791136064-7986c2920216?auto=format&fit=crop&q=80&w=500',
    },
  ];

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-white">SignBridge Blog</h1>
        <p className="mt-2 text-gray-600">Latest insights, stories, and updates from the sign language community</p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {posts.map((post) => (
          <article key={post.id} className="card">
            <img
              src={post.image}
              alt={post.title}
              className="w-full h-48 object-cover rounded-t-xl -mx-6 -mt-6 mb-4"
            />
            <h2 className="text-xl font-semibold mb-2">{post.title}</h2>
            <p className="text-gray-600 mb-4">{post.excerpt}</p>
            <div className="flex items-center justify-between text-sm text-gray-500 mb-4">
              <div className="flex items-center">
                <User className="h-4 w-4 mr-1" />
                {post.author}
              </div>
              <div className="flex items-center">
                <Calendar className="h-4 w-4 mr-1" />
                {post.date}
              </div>
            </div>
            <button className="text-blue-600 font-medium hover:text-blue-700 transition-colors duration-200 flex items-center">
              Read More
              <ArrowRight className="h-4 w-4 ml-1" />
            </button>
          </article>
        ))}
      </div>
    </div>
  );
};

export default Blog;