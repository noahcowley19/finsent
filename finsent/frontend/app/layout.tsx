import type { Metadata } from 'next';
import './globals.css';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';

export const metadata: Metadata = {
  title: 'Caveray | Financial Intelligence Platform',
  description: 'Professional-grade financial analysis tools combining real-time sentiment tracking, academic scoring models, and insider activity monitoring.',
  keywords: ['stock analysis', 'sentiment analysis', 'financial intelligence', 'insider trading', 'portfolio tracker'],
  authors: [{ name: 'Caveray' }],
  openGraph: {
    title: 'Caveray | Financial Intelligence Platform',
    description: 'Professional-grade financial analysis tools for smart investing.',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/* Preconnect to Google Fonts */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        
        {/* Favicon */}
        <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><defs><linearGradient id='g' x1='0%' y1='0%' x2='100%' y2='100%'><stop offset='0%' stop-color='%2300d4aa'/><stop offset='100%' stop-color='%2300a3ff'/></linearGradient></defs><rect width='32' height='32' rx='8' fill='url(%23g)'/><path d='M10 22V14L16 10L22 14V22L16 26L10 22Z' stroke='white' stroke-width='2' fill='none'/><circle cx='16' cy='16' r='3' fill='white'/></svg>" />
        
        {/* Theme color for mobile browsers */}
        <meta name="theme-color" content="#06080d" />
        
        {/* Ads */}
        <script
          async
          src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-5554963041129509"
          crossOrigin="anonymous"
        />
      </head>
      <body>
        <Navbar />
        <main>
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
