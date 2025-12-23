// =============================================================================
// AUTH LAYOUT
// =============================================================================
// Shared layout for authentication pages (sign in, sign up, etc.)
// Provides a clean, centered layout without the main navbar/footer
//
// =============================================================================

import React from 'react';
import Link from 'next/link';

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-cream-50 via-cream-100 to-cream-50 flex flex-col">
      {/* Header with logo */}
      <header className="p-6">
        <Link href="/" className="inline-flex items-center gap-2 group">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-navy-500 to-navy-700 flex items-center justify-center transition-transform duration-fast group-hover:scale-105">
            <svg
              className="w-5 h-5 text-white"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M3 3v18h18" />
              <path d="M18 9l-5 5-4-4-3 3" />
            </svg>
          </div>
          <span className="font-heading font-semibold text-xl text-navy-900">
            Caveray
          </span>
        </Link>
      </header>

      {/* Main content */}
      <main className="flex-1 flex items-center justify-center p-6">
        <div className="w-full max-w-md">
          {children}
        </div>
      </main>

      {/* Footer */}
      <footer className="p-6 text-center">
        <p className="text-body-sm text-neutral-500">
          © {new Date().getFullYear()} Caveray. All rights reserved.
        </p>
      </footer>
    </div>
  );
}
