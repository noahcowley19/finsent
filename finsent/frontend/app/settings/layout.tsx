'use client';

// =============================================================================
// SETTINGS LAYOUT
// =============================================================================
// Shared layout for settings pages with sidebar navigation
//
// Location: frontend/app/settings/layout.tsx
//
// =============================================================================

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Container, Section, AuthGuard } from '@/components/layout';

const settingsNav = [
  {
    label: 'Profile',
    href: '/settings/profile',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
      </svg>
    ),
  },
  {
    label: 'Billing',
    href: '/settings/billing',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
      </svg>
    ),
  },
  {
    label: 'Notifications',
    href: '/settings/notifications',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
      </svg>
    ),
  },
];

export default function SettingsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <AuthGuard>
      {/* Header */}
      <Section spacing="md" background="gradient">
        <h1 className="font-display text-display-sm text-navy-900">Settings</h1>
        <p className="text-body-md text-neutral-600 mt-1">
          Manage your account settings and preferences
        </p>
      </Section>

      {/* Content */}
      <Section spacing="lg" background="default">
        <div className="flex flex-col lg:flex-row gap-8">
          {/* Sidebar navigation */}
          <aside className="lg:w-64 flex-shrink-0">
            <nav className="bg-white rounded-xl border border-border-light p-2 lg:sticky lg:top-24">
              <ul className="space-y-1">
                {settingsNav.map((item) => {
                  const isActive = pathname === item.href;
                  return (
                    <li key={item.href}>
                      <Link
                        href={item.href}
                        className={`
                          flex items-center gap-3 px-4 py-3 rounded-lg
                          text-body-sm font-medium
                          transition-colors duration-fast
                          ${isActive
                            ? 'bg-navy-500/10 text-navy-900'
                            : 'text-neutral-600 hover:bg-cream-50 hover:text-navy-700'
                          }
                        `}
                      >
                        <span className={isActive ? 'text-navy-600' : 'text-neutral-400'}>
                          {item.icon}
                        </span>
                        {item.label}
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </nav>
          </aside>

          {/* Main content */}
          <main className="flex-1 min-w-0">
            {children}
          </main>
        </div>
      </Section>
    </AuthGuard>
  );
}
