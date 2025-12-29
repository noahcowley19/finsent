// =============================================================================
// CAVERAY LAYOUT COMPONENTS
// =============================================================================
// Barrel export file for all layout components.
// 
// Usage:
//   import { Navbar, Footer, Container, Section } from '@/components/layout';
//
// =============================================================================

// -----------------------------------------------------------------------------
// NAVIGATION
// -----------------------------------------------------------------------------
export {
  Navbar,
  navLinks,
  type NavbarProps,
  type NavLink,
  type User,
} from './Navbar';

export {
  MobileMenu,
  type MobileMenuProps,
} from './MobileMenu';

export {
  ConnectedNavbar,
} from './ConnectedNavbar';

export {
  CommandBarNavbar,
} from './CommandBarNavbar';

export {
  NavbarSwitcher,
} from './NavbarSwitcher';

export {
  Footer,
} from './Footer';

// -----------------------------------------------------------------------------
// PAGE STRUCTURE
// -----------------------------------------------------------------------------
export {
  PageHeader,
  PageTitle,
  type PageHeaderProps,
  type PageTitleProps,
  type Breadcrumb,
} from './PageHeader';

export {
  Container,
  NarrowContainer,
  ContentContainer,
  type ContainerProps,
  type ContainerSize,
} from './Container';

export {
  Section,
  SectionHeader,
  HeroSection,
  type SectionProps,
  type SectionBackground,
  type SectionSpacing,
  type SectionHeaderProps,
  type HeroSectionProps,
} from './Section';

// -----------------------------------------------------------------------------
// GRID & LAYOUT
// -----------------------------------------------------------------------------
export {
  Grid,
  GridItem,
  Stack,
  TwoColumn,
  type GridProps,
  type GridCols,
  type GridGap,
  type GridItemProps,
  type StackProps,
  type TwoColumnProps,
} from './Grid';

// -----------------------------------------------------------------------------
// AUTH & FEATURE GATING
// -----------------------------------------------------------------------------
export {
  AuthGuard,
  FeatureGate,
  type AuthGuardProps,
  type AuthGuardUser,
  type UserTier,
  type FeatureGateProps,
} from './AuthGuard';

export {
  UpgradePrompt,
  UpgradeBanner,
  LockedFeatureOverlay,
  type UpgradePromptProps,
  type UpgradeBannerProps,
  type LockedFeatureOverlayProps,
} from './UpgradePrompt';
