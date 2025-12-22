'use client';

import React, { useState, useRef, useEffect, ReactNode, HTMLAttributes, createContext, useContext } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type TabsVariant = 'underline' | 'pill' | 'enclosed';
export type TabsSize = 'sm' | 'md' | 'lg';

export interface TabItem {
  id: string;
  label: string;
  icon?: ReactNode;
  disabled?: boolean;
  badge?: string | number;
}

export interface TabsProps extends HTMLAttributes<HTMLDivElement> {
  /** Currently active tab */
  value: string;
  /** Called when tab changes */
  onChange: (value: string) => void;
  /** Tab variant style */
  variant?: TabsVariant;
  /** Tab size */
  size?: TabsSize;
  /** Full width tabs */
  fullWidth?: boolean;
  /** Children (Tab components) */
  children: ReactNode;
}

export interface TabProps extends HTMLAttributes<HTMLButtonElement> {
  /** Tab identifier */
  value: string;
  /** Tab label */
  label: string;
  /** Icon before label */
  icon?: ReactNode;
  /** Disabled state */
  disabled?: boolean;
  /** Badge content */
  badge?: string | number;
}

export interface TabPanelProps extends HTMLAttributes<HTMLDivElement> {
  /** Tab identifier this panel belongs to */
  value: string;
  /** Panel content */
  children: ReactNode;
}

// =============================================================================
// CONTEXT
// =============================================================================

interface TabsContextValue {
  activeTab: string;
  setActiveTab: (value: string) => void;
  variant: TabsVariant;
  size: TabsSize;
}

const TabsContext = createContext<TabsContextValue | undefined>(undefined);

const useTabsContext = () => {
  const context = useContext(TabsContext);
  if (!context) {
    throw new Error('Tab components must be used within a Tabs component');
  }
  return context;
};

// =============================================================================
// STYLES
// =============================================================================

const sizeStyles: Record<TabsSize, string> = {
  sm: 'text-body-sm py-1.5 px-3',
  md: 'text-body-md py-2 px-4',
  lg: 'text-body-lg py-2.5 px-5',
};

const variantContainerStyles: Record<TabsVariant, string> = {
  underline: 'border-b border-border-light',
  pill: 'bg-cream-100 p-1 rounded-lg',
  enclosed: 'border-b border-border-light',
};

// =============================================================================
// TABS COMPONENT
// =============================================================================

export const Tabs: React.FC<TabsProps> = ({
  value,
  onChange,
  variant = 'underline',
  size = 'md',
  fullWidth = false,
  children,
  className = '',
  ...props
}) => {
  const [indicatorStyle, setIndicatorStyle] = useState({ left: 0, width: 0 });
  const tabsRef = useRef<HTMLDivElement>(null);

  // Update indicator position
  useEffect(() => {
    if (variant !== 'underline' || !tabsRef.current) return;

    const activeTab = tabsRef.current.querySelector(`[data-value="${value}"]`) as HTMLElement;
    if (activeTab) {
      setIndicatorStyle({
        left: activeTab.offsetLeft,
        width: activeTab.offsetWidth,
      });
    }
  }, [value, variant]);

  return (
    <TabsContext.Provider value={{ activeTab: value, setActiveTab: onChange, variant, size }}>
      <div className={className} {...props}>
        {/* Tab list */}
        <div
          ref={tabsRef}
          role="tablist"
          className={`
            relative flex
            ${variantContainerStyles[variant]}
            ${fullWidth ? 'w-full' : 'inline-flex'}
          `}
        >
          {React.Children.map(children, (child) => {
            if (React.isValidElement(child) && child.type === Tab) {
              return React.cloneElement(child as React.ReactElement<TabProps>, {
                className: fullWidth ? 'flex-1' : '',
              });
            }
            return null;
          })}

          {/* Animated underline indicator */}
          {variant === 'underline' && (
            <div
              className="absolute bottom-0 h-0.5 bg-navy-500 transition-all duration-fast ease-out"
              style={{ left: indicatorStyle.left, width: indicatorStyle.width }}
            />
          )}
        </div>

        {/* Tab panels */}
        {React.Children.map(children, (child) => {
          if (React.isValidElement(child) && child.type === TabPanel) {
            return child;
          }
          return null;
        })}
      </div>
    </TabsContext.Provider>
  );
};

// =============================================================================
// TAB COMPONENT
// =============================================================================

export const Tab: React.FC<TabProps> = ({
  value,
  label,
  icon,
  disabled = false,
  badge,
  className = '',
  ...props
}) => {
  const { activeTab, setActiveTab, variant, size } = useTabsContext();
  const isActive = activeTab === value;

  const variantStyles: Record<TabsVariant, { base: string; active: string; inactive: string }> = {
    underline: {
      base: 'relative font-medium transition-colors duration-fast',
      active: 'text-navy-900',
      inactive: 'text-neutral-500 hover:text-navy-700',
    },
    pill: {
      base: 'font-medium rounded-md transition-all duration-fast',
      active: 'bg-white text-navy-900 shadow-sm',
      inactive: 'text-neutral-600 hover:text-navy-900 hover:bg-white/50',
    },
    enclosed: {
      base: 'font-medium border-b-2 -mb-px transition-colors duration-fast',
      active: 'border-navy-500 text-navy-900 bg-white',
      inactive: 'border-transparent text-neutral-500 hover:text-navy-700 hover:border-border-medium',
    },
  };

  const styles = variantStyles[variant];

  return (
    <button
      role="tab"
      type="button"
      data-value={value}
      aria-selected={isActive}
      aria-controls={`panel-${value}`}
      disabled={disabled}
      onClick={() => !disabled && setActiveTab(value)}
      className={`
        ${styles.base}
        ${sizeStyles[size]}
        ${isActive ? styles.active : styles.inactive}
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        inline-flex items-center justify-center gap-2
        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy-500 focus-visible:ring-offset-2
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {icon && <span className="w-4 h-4">{icon}</span>}
      {label}
      {badge !== undefined && (
        <span
          className={`
            ml-1.5 px-1.5 py-0.5
            text-caption font-medium
            rounded-full
            ${isActive ? 'bg-navy-100 text-navy-700' : 'bg-cream-200 text-neutral-600'}
          `}
        >
          {badge}
        </span>
      )}
    </button>
  );
};

// =============================================================================
// TAB PANEL COMPONENT
// =============================================================================

export const TabPanel: React.FC<TabPanelProps> = ({
  value,
  children,
  className = '',
  ...props
}) => {
  const { activeTab } = useTabsContext();
  const isActive = activeTab === value;

  if (!isActive) return null;

  return (
    <div
      role="tabpanel"
      id={`panel-${value}`}
      aria-labelledby={value}
      className={`mt-4 animate-fade-in ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

// =============================================================================
// SIMPLE TABS (convenience component)
// =============================================================================

export interface SimpleTabsProps extends Omit<TabsProps, 'children'> {
  /** Tab items */
  items: TabItem[];
  /** Render content for each tab */
  renderContent: (item: TabItem) => ReactNode;
}

export const SimpleTabs: React.FC<SimpleTabsProps> = ({
  items,
  renderContent,
  ...props
}) => {
  return (
    <Tabs {...props}>
      {items.map((item) => (
        <Tab
          key={item.id}
          value={item.id}
          label={item.label}
          icon={item.icon}
          disabled={item.disabled}
          badge={item.badge}
        />
      ))}
      {items.map((item) => (
        <TabPanel key={item.id} value={item.id}>
          {renderContent(item)}
        </TabPanel>
      ))}
    </Tabs>
  );
};

export default Tabs;
