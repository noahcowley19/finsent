'use client';

// =============================================================================
// CONNECTED NAVBAR
// =============================================================================
// This component connects the Navbar to the authentication system.
// It handles sign in/out navigation and passes user data to the Navbar.
//
// Usage: Just import and use - no props needed
//   <ConnectedNavbar />
//
// =============================================================================

import React from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';
import { Navbar } from './Navbar';

export const ConnectedNavbar: React.FC = () => {
  const router = useRouter();
  const pathname = usePathname();
  const { user, signOut, isLoading } = useAuth();

  // Don't show navbar on auth pages
  const isAuthPage = pathname.startsWith('/signin') || 
                     pathname.startsWith('/signup') || 
                     pathname.startsWith('/forgot-password') ||
                     pathname.startsWith('/reset-password');

  if (isAuthPage) {
    return null;
  }

  // Convert auth user to navbar user format
  const navbarUser = user ? {
    id: user.id,
    email: user.email,
    name: user.name ?? undefined,
    image: user.image ?? undefined,
    tier: user.tier,
  } : null;

  const handleSignIn = () => {
    router.push('/signin');
  };

  const handleSignUp = () => {
    router.push('/signup');
  };

  const handleSignOut = async () => {
    await signOut();
  };

  // Check if current page is a hero page (for transparent navbar)
  const isHeroPage = pathname === '/';

  return (
    <Navbar
      transparent={isHeroPage}
      user={navbarUser}
      onSignIn={handleSignIn}
      onSignUp={handleSignUp}
      onSignOut={handleSignOut}
    />
  );
};

export default ConnectedNavbar;
