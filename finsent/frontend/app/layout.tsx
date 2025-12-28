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
import { AuthProvider } from '@/lib/auth-context';
import { ToastProvider } from '@/components/ui';
import { ConnectedNavbar } from '@/components/layout/ConnectedNavbar';
import { Footer } from '@/components/layout';
import './globals.css';

// =============================================================================
// FONTS
// =============================================================================
// Note: Fonts are loaded via CSS @import in globals.css to avoid build-time
// network requests that may fail in restricted environments

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
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,100..900;1,9..144,100..900&display=swap" rel="stylesheet" />
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,200..800;1,200..800&display=swap" rel="stylesheet" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap" rel="stylesheet" />
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&display=swap" rel="stylesheet" />
      </head>
      <body className="min-h-screen bg-cream-50 font-sans text-obsidian-900 antialiased flex flex-col">
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
