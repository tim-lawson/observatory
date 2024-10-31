'use client';

import { NeuronForDisplay } from '@/app/types/neuronData';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  CaretSortIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  DotsHorizontalIcon,
} from '@radix-ui/react-icons';
import { ColumnDef } from '@tanstack/react-table';

const ROUND_TO = 4;

const locationColumn: ColumnDef<NeuronForDisplay> = {
  accessorFn: (row) => `L${row.layer}/\nN${row.neuron}`,
  id: 'location',
  header: ({ column }) => {
    return (
      <Button
        variant="ghost"
        onClick={() => column.toggleSorting()}
        className="p-0 text-sm font-semibold text-black"
      >
        ID
        {column.getIsSorted() === 'asc' ? (
          <ChevronUpIcon className="ml-0.5 h-4 w-4" />
        ) : column.getIsSorted() === 'desc' ? (
          <ChevronDownIcon className="ml-0.5 h-4 w-4" />
        ) : (
          <CaretSortIcon className="ml-0.5 h-4 w-4" />
        )}
      </Button>
    );
  },
  cell: ({ row }) => <div className="text-xs">{row.getValue('location')}</div>,
};

const activationColumn: ColumnDef<NeuronForDisplay> = {
  accessorKey: 'activation',
  header: ({ column }) => {
    return (
      <Button
        variant="ghost"
        onClick={() => column.toggleSorting()}
        className="p-0 text-sm font-semibold text-black"
      >
        Act
        {column.getIsSorted() === 'asc' ? (
          <ChevronUpIcon className="ml-0.5 h-4 w-4" />
        ) : column.getIsSorted() === 'desc' ? (
          <ChevronDownIcon className="ml-0.5 h-4 w-4" />
        ) : (
          <CaretSortIcon className="ml-0.5 h-4 w-4" />
        )}
      </Button>
    );
  },
  cell: ({ row }) => {
    const activation = row.getValue('activation');
    return (
      <div className="text-xs">
        {typeof activation === 'number' ? activation.toFixed(ROUND_TO) : ''}
      </div>
    );
  },
};

const attributionColumn: ColumnDef<NeuronForDisplay> = {
  accessorKey: 'attribution',
  header: ({ column }) => {
    return (
      <Button
        variant="ghost"
        onClick={() => column.toggleSorting()}
        className="p-0 text-sm font-semibold text-black"
      >
        Attr
        {column.getIsSorted() === 'asc' ? (
          <ChevronUpIcon className="ml-0.5 h-4 w-4" />
        ) : column.getIsSorted() === 'desc' ? (
          <ChevronDownIcon className="ml-0.5 h-4 w-4" />
        ) : (
          <CaretSortIcon className="ml-0.5 h-4 w-4" />
        )}
      </Button>
    );
  },
  cell: ({ row }) => {
    const attribution = row.getValue('attribution');
    return (
      <div className="text-xs">
        {typeof attribution === 'number' ? attribution.toFixed(ROUND_TO) : ''}
      </div>
    );
  },
};

const activationNormalizedColumn: ColumnDef<NeuronForDisplay> = {
  accessorKey: 'activationNormalized',
  header: ({ column }) => {
    return (
      <Button
        variant="ghost"
        onClick={() => column.toggleSorting()}
        className="p-0 text-sm font-semibold text-black"
      >
        Act / Top %ile
        {column.getIsSorted() === 'asc' ? (
          <ChevronUpIcon className="ml-0.5 h-4 w-4" />
        ) : column.getIsSorted() === 'desc' ? (
          <ChevronDownIcon className="ml-0.5 h-4 w-4" />
        ) : (
          <CaretSortIcon className="ml-0.5 h-4 w-4" />
        )}
      </Button>
    );
  },
  cell: ({ row }) => {
    const activation = row.getValue('activationNormalized');
    return (
      <div className="text-xs">
        {typeof activation === 'number' ? activation.toFixed(ROUND_TO) : ''}
      </div>
    );
  },
};

const posDescriptionColumn: ColumnDef<NeuronForDisplay> = {
  accessorKey: 'posDescription',
  header: ({ column }) => {
    return (
      <div className="p-0 text-sm font-semibold text-black">
        (+) Explanation
      </div>
    );
  },
  cell: ({ row }) => (
    <div
      className="text-xs line-clamp-4"
      dangerouslySetInnerHTML={{
        __html:
          (row.getValue('posDescription') as string)?.replace(
            /\{+([^}]+)\}+/g,
            '<u>$1</u>'
          ) || '',
      }}
    />
  ),
};

const negDescriptionColumn: ColumnDef<NeuronForDisplay> = {
  accessorKey: 'negDescription',
  header: ({ column }) => {
    return (
      <div className="p-0 text-sm font-semibold text-black">
        (-) Explanation
      </div>
    );
  },
  cell: ({ row }) => (
    <div
      className="text-xs line-clamp-4"
      dangerouslySetInnerHTML={{
        __html:
          (row.getValue('negDescription') as string)?.replace(
            /\{+([^}]+)\}+/g,
            '<u>$1</u>'
          ) || '',
      }}
    />
  ),
};

const singleExplanationColumn: ColumnDef<NeuronForDisplay> = {
  accessorFn: (row) => row.posDescription || row.negDescription,
  id: 'explanation',
  header: ({ column }) => {
    return (
      <div className="p-0 text-sm font-semibold text-black">Explanation</div>
    );
  },
  cell: ({ row }) => (
    <div
      className="text-xs line-clamp-4"
      dangerouslySetInnerHTML={{
        __html:
          (row.getValue('explanation') as string)?.replace(
            /\{+([^}]+)\}+/g,
            '<u>$1</u>'
          ) || '',
      }}
    />
  ),
};

const scoreColumn: ColumnDef<NeuronForDisplay> = {
  accessorKey: 'score',
  header: ({ column }) => {
    return (
      <Button
        variant="ghost"
        onClick={() => column.toggleSorting()}
        className="p-0 text-sm font-semibold text-black"
      >
        Score
        {column.getIsSorted() === 'asc' ? (
          <ChevronUpIcon className="ml-0.5 h-4 w-4" />
        ) : column.getIsSorted() === 'desc' ? (
          <ChevronDownIcon className="ml-0.5 h-4 w-4" />
        ) : (
          <CaretSortIcon className="ml-0.5 h-4 w-4" />
        )}
      </Button>
    );
  },
  cell: ({ row }) => {
    const score = row.getValue('score');
    return (
      <div className="text-xs">
        {typeof score === 'number' ? score.toFixed(ROUND_TO) : ''}
      </div>
    );
  },
};

const isInterestingColumn: ColumnDef<NeuronForDisplay> = {
  accessorKey: 'isInteresting',
  header: ({ column }) => {
    return (
      <Button
        variant="ghost"
        onClick={() => column.toggleSorting()}
        className="p-0 text-sm font-semibold text-black"
      >
        Interesting
        {column.getIsSorted() === 'asc' ? (
          <ChevronUpIcon className="ml-0.5 h-4 w-4" />
        ) : column.getIsSorted() === 'desc' ? (
          <ChevronDownIcon className="ml-0.5 h-4 w-4" />
        ) : (
          <CaretSortIcon className="ml-0.5 h-4 w-4" />
        )}
      </Button>
    );
  },
  cell: ({ row }) => {
    const isInteresting = row.getValue('isInteresting');
    return <div className="text-xs">{isInteresting ? 'Yes' : 'No'}</div>;
  },
};

export const columns = [
  locationColumn,
  activationColumn,
  posDescriptionColumn,
  negDescriptionColumn,
];
export const columnsSingleExplanation = [
  locationColumn,
  activationNormalizedColumn,
  singleExplanationColumn,
  // scoreColumn,
  // isInterestingColumn,
];
export const columnsAttribution = [
  locationColumn,
  attributionColumn,
  singleExplanationColumn,
  // scoreColumn,
  // isInterestingColumn,
];
export const columnsNoActivation = [
  locationColumn,
  posDescriptionColumn,
  negDescriptionColumn,
];
export const columnsNoActivationSingleExplanation = [
  locationColumn,
  singleExplanationColumn,
  scoreColumn,
];

// {
//   id: 'select',
//   header: ({ table }) => (
//     <Checkbox
//       checked={
//         table.getIsAllPageRowsSelected() ||
//         (table.getIsSomePageRowsSelected() && 'indeterminate')
//       }
//       onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
//       aria-label="Select all"
//       className="p-0 text-sm font-semibold text-black"
//     />
//   ),
//   cell: ({ row }) => (
//     <Checkbox
//       checked={row.getIsSelected()}
//       onCheckedChange={(value) => row.toggleSelected(!!value)}
//       aria-label="Select row"
//     />
//   ),
//   enableSorting: false,
//   enableHiding: false,
// },
