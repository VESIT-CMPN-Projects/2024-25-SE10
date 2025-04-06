import React from "react";
import { Users, MessageSquare, Heart } from "lucide-react";

const Community = () => {
  const discussions = [
    {
      id: 1,
      title: "Tips for Learning Sign Language Faster",
      author: "John Doe",
      replies: 15,
      likes: 32,
      tags: ["Learning", "Tips"],
    },
    {
      id: 2,
      title: "Regional Sign Language Variations Discussion",
      author: "Jane Smith",
      replies: 23,
      likes: 45,
      tags: ["Discussion", "Regional"],
    },
    {
      id: 3,
      title: "Sign Language in Education: Experiences",
      author: "Mike Johnson",
      replies: 18,
      likes: 27,
      tags: ["Education", "Experience"],
    },
  ];

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-white">Community</h1>
        <p className="mt-2 text-gray-600">
          Connect, share, and learn with fellow sign language enthusiasts
        </p>
      </div>

      <div className="flex justify-between items-center">
        <div className="flex space-x-2">
          <button className="btn-primary">Start Discussion</button>
          <button className="nav-link">My Discussions</button>
        </div>
        <div className="relative">
          <input
            type="text"
            placeholder="Search discussions..."
            className="px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      </div>

      <div className="space-y-4">
        {discussions.map((discussion) => (
          <div
            key={discussion.id}
            className="card hover:border-blue-200 cursor-pointer"
          >
            <div className="flex justify-between items-start">
              <div>
                <h3 className="text-xl font-semibold mb-2">
                  {discussion.title}
                </h3>
                <p className="text-gray-600 mb-3">
                  Started by {discussion.author}
                </p>
                <div className="flex space-x-2">
                  {discussion.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-2 py-1 bg-blue-100 text-blue-600 rounded-full text-sm"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
              <div className="flex space-x-4 text-gray-500">
                <div className="flex items-center">
                  <MessageSquare className="h-4 w-4 mr-1" />
                  {discussion.replies}
                </div>
                <div className="flex items-center">
                  <Heart className="h-4 w-4 mr-1" />
                  {discussion.likes}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-blue-50 rounded-xl p-6">
        <h2 className="text-2xl font-semibold mb-4">Community Stats</h2>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="bg-white p-4 rounded-lg">
            <Users className="h-8 w-8 text-blue-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-gray-900">5,234</div>
            <div className="text-gray-600">Members</div>
          </div>
          <div className="bg-white p-4 rounded-lg">
            <MessageSquare className="h-8 w-8 text-blue-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-gray-900">1,423</div>
            <div className="text-gray-600">Discussions</div>
          </div>
          <div className="bg-white p-4 rounded-lg">
            <Heart className="h-8 w-8 text-blue-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-gray-900">8,745</div>
            <div className="text-gray-600">Interactions</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Community;
