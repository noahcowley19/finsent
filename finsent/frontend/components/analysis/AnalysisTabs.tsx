'use client';

// =============================================================================
// ANALYSIS TABS COMPONENT
// =============================================================================
// Navigation tabs for switching between analysis types
//
// Location: frontend/components/analysis/AnalysisTabs.tsx
//
// =============================================================================

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export interface AnalysisTabsProps {
  symbol: string;
}

const tabs = [
  {
    label: 'Overview',
    href: (symbol: string) => `/stock/${symbol}`,
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
      </svg>
    ),
  },
  {
    label: 'Sentiment',
    href: (symbol: string) => `/sentiment/${symbol}`,
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
  {
    label: 'Financials',
    href: (symbol: string) => `/financials/${symbol}`,
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
  },
  {
    label: 'Insider',
    href: (symbol: string) => `/insider/${symbol}`,
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
      </svg>
    ),
  },
];

export const AnalysisTabs: React.FC<AnalysisTabsProps> = ({ symbol }) => {
  const pathname = usePathname();

  return (
    <div className="bg-white border-b border-border-light">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <nav className="flex gap-1 overflow-x-auto py-2 -mb-px">
          {tabs.map((tab) => {
            const href = tab.href(symbol);
            const isActive = pathname === href;

            return (
              <Link
                key={tab.label}
                href={href}
                className={`
                  flex items-center gap-2 px-4 py-3 rounded-lg
                  text-body-sm font-medium whitespace-nowrap
                  transition-colors duration-fast
                  ${isActive
                    ? 'bg-navy-500/10 text-navy-900'
                    : 'text-neutral-600 hover:bg-cream-50 hover:text-navy-700'
                  }
                `}
              >
                <span className={isActive ? 'text-navy-600' : 'text-neutral-400'}>
                  {tab.icon}
                </span>
                {tab.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </div>
  );
};

export default AnalysisTabs;
