'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

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
    <nav className="fixed top-0 left-0 right-0 h-16 bg-card-bg border-b border-border z-[1000] shadow-custom">
      <div className="h-full px-10 flex items-center justify-end">
        <div className="flex gap-2">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`
                px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200
                ${
                  pathname === link.href
                    ? 'text-primary bg-neutral-light'
                    : 'text-secondary hover:text-primary hover:bg-neutral-light'
                }
              `}
            >
              {link.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
