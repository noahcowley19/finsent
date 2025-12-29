'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';
import { MobileMenu } from '@/components/layout/MobileMenu';

// =============================================================================
// TYPES
// =============================================================================

export interface NavLink {
  label: string;
  href: string;
  requiresAuth?: boolean;
  requiresPro?: boolean;
  icon?: React.ReactNode;
}

export interface User {
  id: string;
  email: string;
  name?: string;
  image?: string;
  tier: 'free' | 'pro';
}

export interface NavbarProps {
  transparent?: boolean;
  user?: User | null;
  onSignIn?: () => void;
  onSignUp?: () => void;
  onSignOut?: () => void;
}

// =============================================================================
// NAVIGATION CONFIG
// =============================================================================

export const navLinks: NavLink[] = [
  { label: 'Dashboard', href: '/dashboard' },
  { label: 'Search', href: '/search' },
  { label: 'Sentiment', href: '/sentiment' },
  { label: 'Screener', href: '/screener' },
  { label: 'Movers', href: '/movers' },
  { label: 'Sectors', href: '/sectors' },
  { label: 'Technicals', href: '/technicals' },
  { label: 'Compare', href: '/compare' },
  { label: 'Dividends', href: '/dividends' },
  { label: 'Earnings', href: '/earnings' },
  { label: 'Portfolio', href: '/portfolio' },
  { label: 'Watchlist', href: '/watchlist' },
  { label: 'Journal', href: '/journal' },
  { label: 'Quant Lab', href: '/quant-lab' },
];

const featuredLinks: NavLink[] = [
  { label: 'Search', href: '/search' },
  { label: 'Dashboard', href: '/dashboard' },
  { label: 'Pricing', href: '/pricing' },
];

// =============================================================================
// LOGO COMPONENT
// =============================================================================

const Logo: React.FC = () => {
  return (
    <Link href="/" className="flex items-center gap-2 group">
      <Image
        src="/caveray-wordmark.png"
        alt="Caveray"
        width={140}
        height={32}
        className="h-8 w-auto object-contain transition-opacity duration-150 group-hover:opacity-80"
        priority
      />
    </Link>
  );
};

// =============================================================================
// USER MENU COMPONENT
// =============================================================================

const UserMenu: React.FC<{
  user: User;
  onSignOut?: () => void;
}> = ({ user, onSignOut }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 p-1 rounded-full transition-all duration-150 hover:bg-cream-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-electric-500"
      >
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-obsidian-600 to-obsidian-800 flex items-center justify-center text-white text-sm font-medium">
          {user.name?.charAt(0) || user.email.charAt(0).toUpperCase()}
        </div>
        <svg className={`w-4 h-4 text-obsidian-400 transition-transform duration-150 ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <>
          <div className="fixed inset-0 z-dropdown" onClick={() => setIsOpen(false)} />
          <div className="absolute right-0 mt-2 w-56 bg-white rounded-2xl shadow-diffuse border border-cream-200/50 z-dropdown animate-fade-in-down overflow-hidden">
            <div className="px-4 py-3 border-b border-cream-100">
              <p className="font-semibold text-obsidian-900 truncate">{user.name || 'User'}</p>
              <p className="text-sm text-obsidian-500 truncate">{user.email}</p>
              {user.tier === 'pro' && (
                <span className="inline-flex items-center mt-2 px-2 py-0.5 rounded-full text-xs font-medium bg-electric-100 text-electric-600">
                  Pro
                </span>
              )}
            </div>
            <div className="py-1">
              <Link href="/settings" onClick={() => setIsOpen(false)} className="flex items-center gap-2.5 px-4 py-2.5 text-sm text-obsidian-700 hover:bg-cream-50 transition-colors">
                <svg className="w-4 h-4 text-obsidian-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Settings
              </Link>
              {user.tier === 'free' && (
                <Link href="/pricing" onClick={() => setIsOpen(false)} className="flex items-center gap-2.5 px-4 py-2.5 text-sm text-electric-600 hover:bg-cream-50 transition-colors">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Upgrade
                </Link>
              )}
            </div>
            <div className="border-t border-cream-100 py-1">
              <button onClick={() => { setIsOpen(false); onSignOut?.(); }} className="flex items-center gap-2.5 w-full px-4 py-2.5 text-sm text-obsidian-500 hover:bg-cream-50 transition-colors">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
                Sign out
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

// =============================================================================
// NAVBAR COMPONENT - Cream/Obsidian Aesthetic
// =============================================================================

export const Navbar: React.FC<NavbarProps> = ({
  transparent = false,
  user = null,
  onSignIn,
  onSignUp,
  onSignOut,
}) => {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const pathname = usePathname();

  const handleScroll = useCallback(() => {
    setScrolled(window.scrollY > 10);
  }, []);

  useEffect(() => {
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, [handleScroll]);

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [pathname]);

  const visibleLinks = navLinks.filter((link) => {
    if (link.requiresAuth && !user) return false;
    if (link.requiresPro && user?.tier !== 'pro') return false;
    return true;
  });

  return (
    <>
      <header
        className={`
          fixed top-0 left-0 right-0 z-fixed
          transition-all duration-300 ease-out
          ${scrolled
            ? 'bg-cream-50/90 backdrop-blur-xl border-b border-cream-200/50 shadow-sm'
            : transparent
              ? 'bg-transparent'
              : 'bg-cream-50/80 backdrop-blur-md'
          }
        `}
      >
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className={`flex items-center justify-between transition-all duration-200 ${scrolled ? 'h-14' : 'h-16'}`}>
            <Logo />

            {/* Desktop Navigation */}
            <div className="hidden lg:flex items-center gap-1">
              {featuredLinks.map((link) => {
                const isActive = pathname === link.href || (link.href !== '/' && pathname.startsWith(link.href));
                return (
                  <Link
                    key={link.href}
                    href={link.href}
                    className={`
                      relative px-4 py-2 rounded-xl text-sm font-medium
                      transition-all duration-150
                      ${isActive
                        ? 'text-obsidian-900 bg-cream-200/60'
                        : 'text-obsidian-600 hover:text-obsidian-900 hover:bg-cream-100'
                      }
                    `}
                  >
                    {link.label}
                  </Link>
                );
              })}
            </div>

            {/* Right Side Actions */}
            <div className="flex items-center gap-3">
              {user ? (
                <UserMenu user={user} onSignOut={onSignOut} />
              ) : (
                <div className="hidden sm:flex items-center gap-3">
                  <button
                    onClick={onSignIn}
                    className="px-4 py-2 text-sm font-medium text-obsidian-600 hover:text-obsidian-900 transition-colors"
                  >
                    Log in
                  </button>
                  <button
                    onClick={onSignUp}
                    className="px-5 py-2.5 text-sm font-medium rounded-xl bg-obsidian-900 text-white hover:bg-obsidian-850 transition-all duration-150 hover:-translate-y-0.5 hover:shadow-lg"
                  >
                    Get started
                  </button>
                </div>
              )}

              {/* Mobile Menu Button */}
              <button
                onClick={() => setMobileMenuOpen(true)}
                className="lg:hidden p-2 rounded-xl text-obsidian-600 hover:text-obsidian-900 hover:bg-cream-100 transition-colors"
                aria-label="Open menu"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            </div>
          </div>
        </nav>
      </header>

      <MobileMenu
        isOpen={mobileMenuOpen}
        onClose={() => setMobileMenuOpen(false)}
        links={visibleLinks}
        user={user}
        onSignIn={onSignIn}
        onSignUp={onSignUp}
        onSignOut={onSignOut}
      />

      {/* Spacer */}
      <div className={`${scrolled ? 'h-14' : 'h-16'} transition-all duration-200`} />
    </>
  );
};

export default Navbar;
