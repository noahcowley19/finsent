'use client';

// =============================================================================
// NAVBAR PROXY
// =============================================================================
// This file exists to maintain compatibility with any legacy imports
// and to resolve build issues where the old Navbar path is still being tracked.
//
// It exports the modern, fully-featured Navbar from the layout directory.
// =============================================================================

export { Navbar as default } from './layout/Navbar';
export * from './layout/Navbar';
