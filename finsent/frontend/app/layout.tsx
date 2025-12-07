import type { Metadata, Viewport } from 'next';
import './globals.css';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';

export const metadata: Metadata = {
  title: 'Caveray | Financial Intelligence Platform',
  description: 'Professional-grade financial intelligence combining real-time sentiment analysis, academic scoring models, and insider activity tracking. Make smarter investment decisions.',
  keywords: ['financial analysis', 'sentiment analysis', 'stock research', 'insider trading', 'portfolio management', 'investment tools'],
  authors: [{ name: 'Noah Cowley' }],
  creator: 'Noah Cowley',
  openGraph: {
    title: 'Caveray | Financial Intelligence Platform',
    description: 'Professional-grade financial intelligence for smarter investment decisions.',
    url: 'https://caveray.com',
    siteName: 'Caveray',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Caveray | Financial Intelligence Platform',
    description: 'Professional-grade financial intelligence for smarter investment decisions.',
  },
  icons: {
    icon: [
      {
        url: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40"><defs><linearGradient id="g" x1="0%25" y1="0%25" x2="100%25" y2="100%25"><stop offset="0%25" stop-color="%2300d4aa"/><stop offset="100%25" stop-color="%2300a3ff"/></linearGradient></defs><path d="M20 2L36 11V29L20 38L4 29V11L20 2Z" fill="url(%23g)"/></svg>',
        type: 'image/svg+xml',
      },
    ],
  },
};

export const viewport: Viewport = {
  themeColor: '#030508',
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>
        <Navbar />
        <main>{children}</main>
        <Footer />
      </body>
    </html>
  );
}
