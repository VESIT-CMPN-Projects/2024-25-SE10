import React from 'react';
import { BookOpen, Video, Download, ExternalLink } from 'lucide-react';

const Resources = () => {
  const resources = [
    {
      id: 1,
      title: 'Beginner\'s Guide to Sign Language',
      type: 'PDF Guide',
      description: 'A comprehensive guide covering the basics of sign language.',
      icon: BookOpen,
    },
    {
      id: 2,
      title: 'Sign Language Practice Videos',
      type: 'Video Series',
      description: 'Step-by-step video tutorials for common signs and phrases.',
      icon: Video,
    },
    {
      id: 3,
      title: 'Printable Sign Language Charts',
      type: 'Downloadable',
      description: 'Visual reference charts for alphabet and numbers.',
      icon: Download,
    },
  ];

  const externalLinks = [
    {
      id: 1,
      title: 'National Association of the Deaf',
      url: 'https://www.nad.org',
      description: 'Official resource for deaf and hard of hearing community.',
    },
    {
      id: 2,
      title: 'World Federation of the Deaf',
      url: 'https://wfdeaf.org',
      description: 'International organization supporting deaf rights.',
    },
  ];

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900">Learning Resources</h1>
        <p className="mt-2 text-gray-600">Access our collection of sign language learning materials</p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {resources.map((resource) => (
          <div key={resource.id} className="card">
            <div className="flex items-center space-x-4 mb-4">
              <resource.icon className="h-8 w-8 text-blue-600" />
              <div>
                <h3 className="text-xl font-semibold">{resource.title}</h3>
                <span className="text-sm text-gray-500">{resource.type}</span>
              </div>
            </div>
            <p className="text-gray-600 mb-4">{resource.description}</p>
            <button className="btn-primary w-full">Access Resource</button>
          </div>
        ))}
      </div>

      <div className="bg-blue-50 rounded-xl p-6">
        <h2 className="text-2xl font-semibold mb-4">External Resources</h2>
        <div className="space-y-4">
          {externalLinks.map((link) => (
            <a
              key={link.id}
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block bg-white p-4 rounded-lg hover:shadow-md transition-shadow duration-200"
            >
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-blue-600">{link.title}</h3>
                  <p className="text-gray-600">{link.description}</p>
                </div>
                <ExternalLink className="h-5 w-5 text-gray-400" />
              </div>
            </a>
          ))}
        </div>
      </div>

      <div className="card">
        <h2 className="text-2xl font-semibold mb-4">Need Help?</h2>
        <p className="text-gray-600 mb-4">
          Can't find what you're looking for? Our support team is here to help you
          find the right resources for your learning journey.
        </p>
        <button className="btn-primary">Contact Support</button>
      </div>
    </div>
  );
};

export default Resources;