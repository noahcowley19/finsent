'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';

const navLinks = [
  { href: '/', label: 'Dashboard' },
  { href: '/sentiment', label: 'Sentiment' },
  { href: '/financials', label: 'Financials' },
  { href: '/insider', label: 'Insider Trading' },
  { href: '/search', label: 'Search' },
  { href: '/portfolio', label: 'Portfolio' },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 right-0 h-16 bg-card-bg border-b border-border flex items-center justify-end px-10 z-50 shadow-card">
      <div className="flex gap-2">
        {navLinks.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className={cn(
              'nav-link',
              pathname === link.href && 'active'
            )}
          >
            {link.label}
          </Link>
        ))}
      </div>
    </nav>
  );
}
