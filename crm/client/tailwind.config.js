/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        petrol: {
          DEFAULT: '#00646E',
          50:  '#E6F4F5',
          100: '#CCE9EB',
          200: '#99D3D7',
          300: '#66BDC3',
          400: '#33A7AF',
          500: '#00646E',
          600: '#005059',
          700: '#003C43',
          800: '#00282D',
          900: '#001416',
        },
        siemens: {
          orange: '#EB780A',
          grey:   '#E6E9EB',
          dark:   '#1A1A1A',
        }
      },
      fontFamily: {
        sans: ['Arial', 'Helvetica', 'sans-serif'],
      },
      boxShadow: {
        card: '0 1px 4px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.04)',
      }
    },
  },
  plugins: [],
}
