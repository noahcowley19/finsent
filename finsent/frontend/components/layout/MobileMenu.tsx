'use client';

import React, { useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { NavLink, User } from './Navbar';

// =============================================================================
// TYPES
// =============================================================================

export interface MobileMenuProps {
  /** Whether menu is open */
  isOpen: boolean;
  /** Close handler */
  onClose: () => void;
  /** Navigation links */
  links: NavLink[];
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
// COMPONENT
// =============================================================================

export const MobileMenu: React.FC<MobileMenuProps> = ({
  isOpen,
  onClose,
  links,
  user,
  onSignIn,
  onSignUp,
  onSignOut,
}) => {
  const pathname = usePathname();

  // Lock body scroll when open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
    }
    
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-modal lg:hidden">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-navy-900/50 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Menu panel */}
      <div className="absolute top-0 right-0 h-full w-full max-w-sm bg-cream-50 shadow-2xl animate-slide-in-right">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border-light">
          <Link href="/" onClick={onClose} className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-navy-500 to-navy-700 flex items-center justify-center">
              <svg
                className="w-5 h-5 text-white"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M3 3v18h18" />
                <path d="M18 9l-5 5-4-4-3 3" />
              </svg>
            </div>
            <span className="font-heading font-semibold text-xl text-navy-900">
              Caveray
            </span>
          </Link>

          <button
            onClick={onClose}
            className="p-2 rounded-lg text-neutral-500 hover:text-navy-900 hover:bg-cream-100 transition-colors"
            aria-label="Close menu"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* User info (if logged in) */}
        {user && (
          <div className="p-4 border-b border-border-light bg-cream-100/50">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-terra-400 to-terra-600 flex items-center justify-center text-white font-medium">
                {user.name?.charAt(0) || user.email.charAt(0).toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-body-sm font-medium text-navy-900 truncate">
                  {user.name || 'User'}
                </p>
                <p className="text-caption text-neutral-500 truncate">
                  {user.email}
                </p>
              </div>
              {user.tier === 'pro' && (
                <span className="px-2 py-0.5 rounded-full text-caption font-medium bg-terra-100 text-terra-700">
                  Pro
                </span>
              )}
            </div>
          </div>
        )}

        {/* Navigation links */}
        <nav className="p-4">
          <ul className="space-y-1">
            {links.map((link, index) => {
              const isActive = pathname === link.href || 
                (link.href !== '/' && pathname.startsWith(link.href));

              return (
                <li 
                  key={link.href}
                  className="animate-fade-in-up"
                  style={{ animationDelay: `${index * 50}ms` }}
                >
                  <Link
                    href={link.href}
                    onClick={onClose}
                    className={`
                      flex items-center gap-3 px-4 py-3 rounded-lg
                      text-body-md font-medium
                      transition-colors duration-fast
                      ${isActive 
                        ? 'bg-navy-500/10 text-navy-900' 
                        : 'text-navy-700 hover:bg-cream-100'
                      }
                    `}
                  >
                    {link.icon && (
                      <span className="w-5 h-5 text-neutral-400">
                        {link.icon}
                      </span>
                    )}
                    <span className="flex-1">{link.label}</span>
                    {link.requiresPro && (
                      <span className="px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider bg-terra-500 text-white rounded">
                        Pro
                      </span>
                    )}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Footer actions */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-border-light bg-cream-50">
          {user ? (
            <div className="space-y-2">
              <Link
                href="/settings"
                onClick={onClose}
                className="flex items-center justify-center gap-2 w-full px-4 py-3 rounded-lg border border-border-medium text-navy-700 font-medium hover:bg-cream-100 transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Account Settings
              </Link>
              <button
                onClick={() => {
                  onClose();
                  onSignOut?.();
                }}
                className="flex items-center justify-center gap-2 w-full px-4 py-3 rounded-lg text-neutral-600 font-medium hover:bg-cream-100 transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
                Sign Out
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              <button
                onClick={() => {
                  onClose();
                  onSignUp?.();
                }}
                className="w-full px-4 py-3 rounded-lg bg-terra-500 text-white font-medium hover:bg-terra-600 transition-colors"
              >
                Get Started
              </button>
              <button
                onClick={() => {
                  onClose();
                  onSignIn?.();
                }}
                className="w-full px-4 py-3 rounded-lg border border-border-medium text-navy-700 font-medium hover:bg-cream-100 transition-colors"
              >
                Sign In
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MobileMenu;
