'use client';

import Link from 'next/link';
import Image from 'next/image';
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
            {/* Logo Image */}
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              overflow: 'hidden',
              background: 'linear-gradient(135deg, #00d4aa 0%, #00a3ff 100%)',
              padding: '2px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <Image 
                src="/logo.png" 
                alt="Caveray" 
                width={28} 
                height={28}
                style={{ 
                  objectFit: 'contain',
                  filter: 'brightness(1.1)',
                }}
              />
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
