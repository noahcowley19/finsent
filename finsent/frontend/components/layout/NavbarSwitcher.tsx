'use client';

// =============================================================================
// NAVBAR SWITCHER
// =============================================================================
// Conditionally renders CommandBarNavbar on landing page, ConnectedNavbar elsewhere
//
// Location: frontend/components/layout/NavbarSwitcher.tsx
// =============================================================================

import React from 'react';
import { usePathname } from 'next/navigation';
import { ConnectedNavbar } from './ConnectedNavbar';
import { CommandBarNavbar } from './CommandBarNavbar';

export const NavbarSwitcher: React.FC = () => {
    const pathname = usePathname();

    // Use Command Bar on landing page
    if (pathname === '/') {
        return <CommandBarNavbar />;
    }

    // Use standard navbar elsewhere
    return <ConnectedNavbar />;
};

export default NavbarSwitcher;
