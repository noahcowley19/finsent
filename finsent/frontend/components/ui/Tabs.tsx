'use client';

import React, { createContext, useContext, useState, ReactNode, HTMLAttributes } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type TabsVariant = 'line' | 'pills' | 'enclosed';
export type TabsSize = 'sm' | 'md';

interface TabsContextValue {
  activeTab: string;
  setActiveTab: (id: string) => void;
  variant: TabsVariant;
  size: TabsSize;
}

// =============================================================================
// CONTEXT
// =============================================================================

const TabsContext = createContext<TabsContextValue | null>(null);

const useTabsContext = () => {
  const context = useContext(TabsContext);
  if (!context) {
    throw new Error('Tab components must be used within a Tabs component');
  }
  return context;
};

// =============================================================================
// TABS COMPONENT
// =============================================================================

export interface TabsProps extends Omit<HTMLAttributes<HTMLDivElement>, 'onChange'> {
  /** Default active tab */
  defaultTab: string;
  /** Controlled active tab */
  activeTab?: string;
  /** Change handler */
  onChange?: (tabId: string) => void;
  /** Visual variant */
  variant?: TabsVariant;
  /** Tab size */
  size?: TabsSize;
  /** Content */
  children: ReactNode;
}

export const Tabs: React.FC<TabsProps> = ({
  defaultTab,
  activeTab: controlledActiveTab,
  onChange,
  variant = 'line',
  size = 'md',
  children,
  className = '',
  ...props
}) => {
  const [internalActiveTab, setInternalActiveTab] = useState(defaultTab);
  const activeTab = controlledActiveTab ?? internalActiveTab;

  const setActiveTab = (id: string) => {
    if (!controlledActiveTab) {
      setInternalActiveTab(id);
    }
    onChange?.(id);
  };

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab, variant, size }}>
      <div className={className} {...props}>
        {children}
      </div>
    </TabsContext.Provider>
  );
};

// =============================================================================
// TAB LIST
// =============================================================================

export interface TabListProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

const variantListStyles: Record<TabsVariant, string> = {
  line: 'border-b border-ink-200',
  pills: 'bg-ink-100 p-1 rounded-xl',
  enclosed: 'bg-ink-50 p-1 rounded-xl',
};

export const TabList: React.FC<TabListProps> = ({
  children,
  className = '',
  ...props
}) => {
  const { variant } = useTabsContext();

  return (
    <div
      role="tablist"
      className={`
        flex
        ${variantListStyles[variant]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {children}
    </div>
  );
};

// =============================================================================
// TAB
// =============================================================================

export interface TabProps extends HTMLAttributes<HTMLButtonElement> {
  /** Tab identifier */
  id: string;
  /** Disabled state */
  disabled?: boolean;
  /** Content */
  children: ReactNode;
}

const sizeStyles: Record<TabsSize, string> = {
  sm: 'px-3 py-1.5 text-body-sm',
  md: 'px-4 py-2 text-body-sm',
};

const getTabStyles = (variant: TabsVariant, isActive: boolean, disabled: boolean) => {
  if (disabled) {
    return 'opacity-50 cursor-not-allowed';
  }

  const baseStyles = 'font-medium transition-all duration-150';

  switch (variant) {
    case 'line':
      return `${baseStyles} -mb-px border-b-2 ${isActive
          ? 'text-ink-900 border-ink-900'
          : 'text-ink-500 border-transparent hover:text-ink-700 hover:border-ink-300'
        }`;
    case 'pills':
      return `${baseStyles} rounded-lg ${isActive
          ? 'bg-white text-ink-900 shadow-sm'
          : 'text-ink-600 hover:text-ink-900 hover:bg-white/50'
        }`;
    case 'enclosed':
      return `${baseStyles} rounded-lg ${isActive
          ? 'bg-white text-ink-900 shadow-sm'
          : 'text-ink-500 hover:text-ink-700'
        }`;
    default:
      return baseStyles;
  }
};

export const Tab: React.FC<TabProps> = ({
  id,
  disabled = false,
  children,
  className = '',
  ...props
}) => {
  const { activeTab, setActiveTab, variant, size } = useTabsContext();
  const isActive = activeTab === id;

  return (
    <button
      role="tab"
      aria-selected={isActive}
      aria-controls={`panel-${id}`}
      id={`tab-${id}`}
      disabled={disabled}
      onClick={() => !disabled && setActiveTab(id)}
      className={`
        ${sizeStyles[size]}
        ${getTabStyles(variant, isActive, disabled)}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {children}
    </button>
  );
};

// =============================================================================
// TAB PANELS
// =============================================================================

export interface TabPanelsProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

export const TabPanels: React.FC<TabPanelsProps> = ({
  children,
  className = '',
  ...props
}) => {
  return (
    <div className={className} {...props}>
      {children}
    </div>
  );
};

// =============================================================================
// TAB PANEL
// =============================================================================

export interface TabPanelProps extends HTMLAttributes<HTMLDivElement> {
  /** Tab identifier this panel belongs to */
  id: string;
  /** Content */
  children: ReactNode;
}

export const TabPanel: React.FC<TabPanelProps> = ({
  id,
  children,
  className = '',
  ...props
}) => {
  const { activeTab } = useTabsContext();
  const isActive = activeTab === id;

  if (!isActive) return null;

  return (
    <div
      role="tabpanel"
      id={`panel-${id}`}
      aria-labelledby={`tab-${id}`}
      className={`animate-fade-in ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

export default Tabs;
