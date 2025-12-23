'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { MobileMenu } from './MobileMenu';

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
  /** Transparent background (for hero sections) */
  transparent?: boolean;
  /** Current user */
  user?: User | null;
  /** Sign in handler */
  onSignIn?: () => void;
  /** Sign up handler */
  onSignUp?: () => void;
  /** Sign out handler */
  onSignOut?: () => void;
}

// =============================================================================
// NAVIGATION CONFIG
// =============================================================================

export const navLinks: NavLink[] = [
  { label: 'Dashboard', href: '/' },
  { label: 'Search', href: '/search' },
  { label: 'Sentiment', href: '/sentiment', requiresAuth: true },
  { label: 'Financials', href: '/financials', requiresAuth: true },
  { label: 'Insider', href: '/insider', requiresAuth: true },
  { label: 'Portfolio', href: '/portfolio', requiresAuth: true },
  { label: 'Quant Lab', href: '/quant-lab', requiresAuth: true, requiresPro: true },
];

// =============================================================================
// LOGO COMPONENT
// =============================================================================

const Logo: React.FC<{ scrolled?: boolean; transparent?: boolean }> = ({ 
  scrolled, 
  transparent 
}) => {
  const textColor = transparent && !scrolled ? 'text-white' : 'text-navy-900';
  
  return (
    <Link href="/" className="flex items-center gap-2 group">
      {/* Logo mark */}
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
      {/* Wordmark */}
      <span className={`font-heading font-semibold text-xl ${textColor} transition-colors duration-fast`}>
        Caveray
      </span>
    </Link>
  );
};

// =============================================================================
// USER MENU COMPONENT
// =============================================================================

const UserMenu: React.FC<{
  user: User;
  onSignOut?: () => void;
  scrolled?: boolean;
  transparent?: boolean;
}> = ({ user, onSignOut, scrolled, transparent }) => {
  const [isOpen, setIsOpen] = useState(false);
  const textColor = transparent && !scrolled ? 'text-white' : 'text-navy-700';

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`
          flex items-center gap-2 p-1.5 rounded-lg
          transition-colors duration-fast
          hover:bg-navy-500/10
          focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy-500
        `}
      >
        {/* Avatar */}
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-terra-400 to-terra-600 flex items-center justify-center text-white text-sm font-medium">
          {user.name?.charAt(0) || user.email.charAt(0).toUpperCase()}
        </div>
        {/* Dropdown arrow */}
        <svg
          className={`w-4 h-4 ${textColor} transition-transform duration-fast ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Dropdown */}
      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-dropdown"
            onClick={() => setIsOpen(false)}
          />
          
          {/* Menu */}
          <div className="absolute right-0 mt-2 w-56 bg-white rounded-lg shadow-xl border border-border-light z-dropdown animate-fade-in-down">
            {/* User info */}
            <div className="px-4 py-3 border-b border-border-light">
              <p className="text-body-sm font-medium text-navy-900 truncate">
                {user.name || 'User'}
              </p>
              <p className="text-caption text-neutral-500 truncate">
                {user.email}
              </p>
              {user.tier === 'pro' && (
                <span className="inline-flex items-center mt-1.5 px-2 py-0.5 rounded-full text-caption font-medium bg-terra-100 text-terra-700">
                  Pro
                </span>
              )}
            </div>

            {/* Menu items */}
            <div className="py-1">
              <Link
                href="/settings"
                onClick={() => setIsOpen(false)}
                className="flex items-center gap-3 px-4 py-2 text-body-sm text-navy-700 hover:bg-cream-50 transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Account Settings
              </Link>
              
              {user.tier === 'free' && (
                <Link
                  href="/pricing"
                  onClick={() => setIsOpen(false)}
                  className="flex items-center gap-3 px-4 py-2 text-body-sm text-terra-600 hover:bg-cream-50 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Upgrade to Pro
                </Link>
              )}
            </div>

            {/* Sign out */}
            <div className="border-t border-border-light py-1">
              <button
                onClick={() => {
                  setIsOpen(false);
                  onSignOut?.();
                }}
                className="flex items-center gap-3 w-full px-4 py-2 text-body-sm text-neutral-600 hover:bg-cream-50 transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
                Sign Out
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

// =============================================================================
// NAVBAR COMPONENT
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

  // Handle scroll
  const handleScroll = useCallback(() => {
    setScrolled(window.scrollY > 20);
  }, []);

  useEffect(() => {
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll(); // Check initial state
    return () => window.removeEventListener('scroll', handleScroll);
  }, [handleScroll]);

  // Close mobile menu on route change
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [pathname]);

  // Filter visible links based on auth state
  const visibleLinks = navLinks.filter((link) => {
    if (link.requiresAuth && !user) return false;
    if (link.requiresPro && user?.tier !== 'pro') return false;
    return true;
  });

  // Dynamic styles
  const bgClass = transparent && !scrolled
    ? 'bg-transparent'
    : 'bg-cream-50/95 backdrop-blur-md shadow-sm';
  
  const textColor = transparent && !scrolled ? 'text-white' : 'text-navy-700';
  const activeTextColor = transparent && !scrolled ? 'text-white' : 'text-navy-900';

  return (
    <>
      <header
        className={`
          fixed top-0 left-0 right-0 z-fixed
          transition-all duration-normal ease-out
          ${bgClass}
        `}
      >
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className={`
            flex items-center justify-between
            transition-all duration-normal
            ${scrolled ? 'h-16' : 'h-20'}
          `}>
            {/* Logo */}
            <Logo scrolled={scrolled} transparent={transparent} />

            {/* Desktop Navigation */}
            <div className="hidden lg:flex items-center gap-1">
              {visibleLinks.map((link) => {
                const isActive = pathname === link.href || 
                  (link.href !== '/' && pathname.startsWith(link.href));

                return (
                  <Link
                    key={link.href}
                    href={link.href}
                    className={`
                      relative px-4 py-2 rounded-lg
                      text-body-sm font-medium
                      transition-all duration-fast
                      ${isActive 
                        ? `${activeTextColor} bg-navy-500/10` 
                        : `${textColor} hover:bg-navy-500/5`
                      }
                    `}
                  >
                    {link.label}
                    {link.requiresPro && (
                      <span className="ml-1.5 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider bg-terra-500 text-white rounded">
                        Pro
                      </span>
                    )}
                  </Link>
                );
              })}
            </div>

            {/* Right section */}
            <div className="flex items-center gap-3">
              {user ? (
                <UserMenu 
                  user={user} 
                  onSignOut={onSignOut}
                  scrolled={scrolled}
                  transparent={transparent}
                />
              ) : (
                <div className="hidden sm:flex items-center gap-3">
                  <button
                    onClick={onSignIn}
                    className={`
                      px-4 py-2 text-body-sm font-medium rounded-lg
                      transition-colors duration-fast
                      ${textColor} hover:bg-navy-500/10
                    `}
                  >
                    Sign In
                  </button>
                  <button
                    onClick={onSignUp}
                    className="
                      px-4 py-2 text-body-sm font-medium rounded-lg
                      bg-terra-500 text-white
                      hover:bg-terra-600
                      transition-colors duration-fast
                      shadow-sm hover:shadow-terra
                    "
                  >
                    Get Started
                  </button>
                </div>
              )}

              {/* Mobile menu button */}
              <button
                onClick={() => setMobileMenuOpen(true)}
                className={`
                  lg:hidden p-2 rounded-lg
                  transition-colors duration-fast
                  ${textColor} hover:bg-navy-500/10
                `}
                aria-label="Open menu"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            </div>
          </div>
        </nav>
      </header>

      {/* Mobile menu */}
      <MobileMenu
        isOpen={mobileMenuOpen}
        onClose={() => setMobileMenuOpen(false)}
        links={visibleLinks}
        user={user}
        onSignIn={onSignIn}
        onSignUp={onSignUp}
        onSignOut={onSignOut}
      />

      {/* Spacer to prevent content from going under fixed navbar */}
      <div className={`${scrolled ? 'h-16' : 'h-20'} transition-all duration-normal`} />
    </>
  );
};

export default Navbar;
