// =============================================================================
// ROOT LAYOUT
// =============================================================================
// Main application layout with all providers and navigation
//
// This file wraps the entire application with:
// - AuthProvider (authentication state)
// - ToastProvider (notifications)
// - Navbar (navigation with auth integration)
// - Footer (site footer)
//
// =============================================================================

import type { Metadata } from 'next';
import { Fraunces, Plus_Jakarta_Sans, Inter } from 'next/font/google';
import { AuthProvider } from '@/lib/auth-context';
import { ToastProvider } from '@/components/ui';
import { ConnectedNavbar } from '@/components/layout/ConnectedNavbar';
import { Footer } from '@/components/layout';
import './globals.css';

// =============================================================================
// FONTS
// =============================================================================

const fraunces = Fraunces({
  subsets: ['latin'],
  variable: '--font-fraunces',
  display: 'swap',
});

const plusJakarta = Plus_Jakarta_Sans({
  subsets: ['latin'],
  variable: '--font-plus-jakarta',
  display: 'swap',
});

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

// =============================================================================
// METADATA
// =============================================================================

export const metadata: Metadata = {
  title: {
    default: 'Caveray - Financial Intelligence Platform',
    template: '%s | Caveray',
  },
  description:
    'AI-powered financial intelligence platform. Analyze market sentiment, track insider trading, and make smarter investment decisions.',
  keywords: [
    'stock analysis',
    'sentiment analysis',
    'financial intelligence',
    'insider trading',
    'market analysis',
    'AI investing',
  ],
  authors: [{ name: 'Caveray' }],
  creator: 'Caveray',
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://caveray.com',
    siteName: 'Caveray',
    title: 'Caveray - Financial Intelligence Platform',
    description:
      'AI-powered financial intelligence platform. Analyze market sentiment, track insider trading, and make smarter investment decisions.',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Caveray - Financial Intelligence Platform',
    description:
      'AI-powered financial intelligence platform. Analyze market sentiment, track insider trading, and make smarter investment decisions.',
  },
  robots: {
    index: true,
    follow: true,
  },
};

// =============================================================================
// LAYOUT
// =============================================================================

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${fraunces.variable} ${plusJakarta.variable} ${inter.variable}`}
    >
      <body className="min-h-screen bg-cream-50 font-body text-navy-900 antialiased flex flex-col">
        <AuthProvider>
          <ToastProvider position="top-right">
            {/* Navigation */}
            <ConnectedNavbar />

            {/* Main content */}
            <main className="flex-1">{children}</main>

            {/* Footer */}
            <Footer />
          </ToastProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
