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
      className="navbar"
      style={{
        borderBottom: scrolled ? '1px solid var(--border)' : '1px solid transparent',
        background: scrolled 
          ? 'rgba(6, 8, 13, 0.9)' 
          : 'rgba(6, 8, 13, 0.6)',
        transition: 'all 0.3s ease',
      }}
    >
      {/* Logo */}
      <Link href="/" className="nav-logo" style={{ textDecoration: 'none' }}>
        <span style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '8px',
        }}>
          {/* Logo Icon */}
          <svg 
            width="28" 
            height="28" 
            viewBox="0 0 32 32" 
            fill="none"
            style={{ flexShrink: 0 }}
          >
            <rect 
              x="2" 
              y="2" 
              width="28" 
              height="28" 
              rx="8" 
              fill="url(#logoGradient)"
            />
            <path 
              d="M10 22V14L16 10L22 14V22L16 26L10 22Z" 
              stroke="white" 
              strokeWidth="2" 
              strokeLinejoin="round"
              fill="none"
            />
            <circle cx="16" cy="16" r="3" fill="white" />
            <defs>
              <linearGradient id="logoGradient" x1="2" y1="2" x2="30" y2="30" gradientUnits="userSpaceOnUse">
                <stop stopColor="#00d4aa" />
                <stop offset="1" stopColor="#00a3ff" />
              </linearGradient>
            </defs>
          </svg>
          
          {/* Logo Text */}
          <span style={{
            fontSize: '1.375rem',
            fontWeight: 800,
            letterSpacing: '-0.03em',
            background: 'linear-gradient(135deg, #00d4aa 0%, #00a3ff 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}>
            Caveray
          </span>
        </span>
      </Link>

      {/* Navigation Links */}
      <div className="nav-links">
        {navLinks.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className={`nav-link ${pathname === link.href ? 'active' : ''}`}
          >
            {link.label}
          </Link>
        ))}
      </div>
    </nav>
  );
}
