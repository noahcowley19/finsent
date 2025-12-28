// =============================================================================
// CAVERAY UI COMPONENT LIBRARY
// =============================================================================
// Barrel export file for all UI components.
// 
// Usage:
//   import { Button, Card, Input, Badge } from '@/components/ui';
//
// =============================================================================

// -----------------------------------------------------------------------------
// BUTTONS
// -----------------------------------------------------------------------------
export {
  Button,
  IconButton,
  ButtonGroup,
  type ButtonProps,
  type ButtonVariant,
  type ButtonSize,
  type IconButtonProps,
  type ButtonGroupProps,
} from './Button';

// -----------------------------------------------------------------------------
// FORM INPUTS
// -----------------------------------------------------------------------------
export {
  Input,
  PasswordInput,
  type InputProps,
  type InputSize,
  type PasswordInputProps,
} from './Input';

export {
  Textarea,
  type TextareaProps,
} from './Textarea';

export {
  Select,
  type SelectProps,
  type SelectSize,
  type SelectOption,
} from './Select';

export {
  SearchInput,
  type SearchInputProps,
  type SearchInputSize,
} from './SearchInput';

// -----------------------------------------------------------------------------
// CARDS
// -----------------------------------------------------------------------------
export {
  Card,
  CardHeader,
  CardBody,
  CardFooter,
  MetricCard,
  FeatureCard,
  type CardProps,
  type CardVariant,
  type CardPadding,
  type CardHeaderProps,
  type CardBodyProps,
  type CardFooterProps,
  type MetricCardProps,
  type FeatureCardProps,
} from './Card';

// -----------------------------------------------------------------------------
// BADGES & TAGS
// -----------------------------------------------------------------------------
export {
  Badge,
  StatusBadge,
  CountBadge,
  type BadgeProps,
  type BadgeVariant,
  type BadgeSize,
  type StatusBadgeProps,
  type CountBadgeProps,
} from './Badge';

export {
  Tag,
  TagGroup,
  type TagProps,
  type TagVariant,
  type TagSize,
  type TagGroupProps,
} from './Tag';

// -----------------------------------------------------------------------------
// LOADING STATES
// -----------------------------------------------------------------------------
export {
  Spinner,
  LoadingDots,
  PageSpinner,
  type SpinnerProps,
  type SpinnerSize,
  type SpinnerVariant,
  type LoadingDotsProps,
  type PageSpinnerProps,
} from './Spinner';

export {
  Skeleton,
  SkeletonText,
  SkeletonAvatar,
  SkeletonCard,
  SkeletonTable,
  type SkeletonProps,
  type SkeletonTextProps,
  type SkeletonAvatarProps,
  type SkeletonCardProps,
  type SkeletonTableProps,
} from './Skeleton';

export {
  Progress,
  CircularProgress,
  UsageIndicator,
  type ProgressProps,
  type ProgressSize,
  type ProgressVariant,
  type CircularProgressProps,
  type UsageIndicatorProps,
} from './Progress';

// -----------------------------------------------------------------------------
// OVERLAYS
// -----------------------------------------------------------------------------
export {
  Modal,
  ModalFooter,
  ConfirmModal,
  type ModalProps,
  type ModalSize,
  type ModalFooterProps,
  type ConfirmModalProps,
} from './Modal';

export {
  Drawer,
  DrawerFooter,
  type DrawerProps,
  type DrawerPosition,
  type DrawerSize,
  type DrawerFooterProps,
} from './Drawer';

export {
  ToastProvider,
  useToast,
  useToastActions,
  type Toast,
  type ToastVariant,
  type ToastPosition,
  type ToastProps,
  type ToastProviderProps,
  type ToastContextValue,
} from './Toast';

export {
  Tooltip,
  InfoTooltip,
  type TooltipProps,
  type TooltipPosition,
  type InfoTooltipProps,
} from './Tooltip';

// -----------------------------------------------------------------------------
// NAVIGATION
// -----------------------------------------------------------------------------
export {
  Tabs,
  Tab,
  TabPanel,
  SimpleTabs,
  type TabsProps,
  type TabsVariant,
  type TabsSize,
  type TabProps,
  type TabPanelProps,
  type TabItem,
  type SimpleTabsProps,
} from './Tabs';

// -----------------------------------------------------------------------------
// LAYOUT
// -----------------------------------------------------------------------------
export {
  Divider,
  OrDivider,
  type DividerProps,
  type DividerOrientation,
  type DividerVariant,
} from './Divider';

// -----------------------------------------------------------------------------
// ANIMATION & EFFECTS
// -----------------------------------------------------------------------------
export { AnimatedCounter, type default as AnimatedCounterDefault } from './AnimatedCounter';
export { GlassCard, GlassPanel } from './GlassCard';
export { MeshGradient, type default as MeshGradientDefault } from './MeshGradient';
