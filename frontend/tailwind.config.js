/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#4f46e5',
          light: '#6366f1',
          dark: '#3730a3'
        },
        secondary: {
          DEFAULT: '#8b5cf6',
          light: '#a78bfa',
          dark: '#6d28d9'
        },
        accent: {
          DEFAULT: '#ec4899',
          light: '#f472b6',
          dark: '#be185d'
        },
        dark: {
          bg: '#121927',
          card: '#1e293b',
          muted: '#334155'
        }
      },
      animation: {
        'gradient-x': 'gradient-x 3s ease infinite',
        'gradient-y': 'gradient-y 3s ease infinite',
        'gradient-xy': 'gradient-xy 3s ease infinite',
        'blob': 'blob 10s infinite',
      },
      keyframes: {
        'gradient-x': {
          '0%, 100%': {
            'background-size': '200% 200%',
            'background-position': 'left center'
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'right center'
          }
        },
        'gradient-y': {
          '0%, 100%': {
            'background-size': '200% 200%',
            'background-position': 'top center'
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'bottom center'
          }
        },
        'gradient-xy': {
          '0%, 100%': {
            'background-size': '400% 400%',
            'background-position': 'left top'
          },
          '25%': {
            'background-position': 'right top'
          },
          '50%': {
            'background-position': 'right bottom'
          },
          '75%': {
            'background-position': 'left bottom'
          }
        },
        'blob': {
          '0%': {
            transform: 'scale(1) translate(0px, 0px)'
          },
          '33%': {
            transform: 'scale(1.1) translate(30px, -50px)'
          },
          '66%': {
            transform: 'scale(0.9) translate(-20px, 20px)'
          },
          '100%': {
            transform: 'scale(1) translate(0px, 0px)'
          }
        }
      },
      transitionProperty: {
        'height': 'height',
        'spacing': 'margin, padding',
      },
      backdropBlur: {
        xs: '2px',
      },
      backgroundImage: {
        'dot-pattern': "radial-gradient(rgba(99, 102, 241, 0.1) 1px, transparent 1px)",
        'dot-pattern-dark': "radial-gradient(rgba(167, 139, 250, 0.15) 1px, transparent 1px)",
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
      backgroundSize: {
        'dot-sm': '20px 20px',
        'dot-md': '30px 30px',
      },
    },
    // Add animation delay utilities
    animationDelay: {
      2000: '2s',
      4000: '4s',
      6000: '6s',
    },
  },
  plugins: [
    function({ addUtilities, theme }) {
      const newUtilities = {};
      Object.entries(theme('animationDelay')).forEach(([key, value]) => {
        newUtilities[`.animation-delay-${key}`] = { animationDelay: value };
      });
      addUtilities(newUtilities);
    }
  ],
};
