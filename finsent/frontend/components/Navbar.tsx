'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState, useEffect } from 'react';

const navLinks = [
  { href: '/', label: 'Dashboard' },
  { href: '/search', label: 'Search' },
  { href: '/sentiment', label: 'Sentiment' },
  { href: '/financials', label: 'Financials' },
  { href: '/insider', label: 'Insider' },
  { href: '/portfolio', label: 'Portfolio' },
];

export default function Navbar() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav 
      className="navbar-glass"
      style={{
        transform: scrolled ? 'translateY(0)' : 'translateY(0)',
        boxShadow: scrolled 
          ? '0 8px 32px rgba(0, 0, 0, 0.4), 0 0 1px rgba(255, 255, 255, 0.1)' 
          : '0 4px 24px rgba(0, 0, 0, 0.2)',
      }}
    >
      <div className="navbar-glass-container">
        {/* Logo */}
        <Link href="/" className="nav-logo-glass" style={{ textDecoration: 'none' }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '10px',
          }}>
            {/* Logo SVG */}
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              background: 'linear-gradient(135deg, #00d4aa 0%, #00a3ff 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            }}>
              <svg width="20" height="20" viewBox="0 0 32 32" fill="none">
                <path d="M10 22V14L16 10L22 14V22L16 26L10 22Z" stroke="white" strokeWidth="2.5" strokeLinejoin="round" />
                <circle cx="16" cy="16" r="3" fill="white" />
              </svg>
            </div>
            
            {/* Logo Text */}
            <span style={{
              fontSize: '1.375rem',
              fontWeight: 800,
              letterSpacing: '-0.03em',
              background: 'linear-gradient(135deg, #00d4aa 0%, #00a3ff 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              whiteSpace: 'nowrap',
            }}>
              Caveray
            </span>
          </div>
        </Link>

        {/* Navigation Links */}
        <div className="nav-links-glass">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`nav-link-glass ${pathname === link.href ? 'active' : ''}`}
            >
              {link.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
